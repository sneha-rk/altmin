require 'torch'
require 'dataset-mnist'
require 'nn'
require 'image'
require 'os'
require 'gnuplot'
require 'optim'
-- signal = require('posix.signal')

opt = 
{
	epochs = 2,
	batch_size = 50,
	print_every = 250,  
	train_size = 60000,
	test_size = 1000,
	-- Learning Rate eta = eps/kappa. kappa = exp(W), where ||w*|| < W Let eps = 1e-2; W ~ 10?
	epsilon = 1e-1,
	w = 0, --Recheck. This slows stuff down!
}



--Our major problem!
opt.learningRate = opt.epsilon * torch.exp(-opt.w)
-- opt.learningRate = opt.epsilon
print(opt)

sgd_params = {
   learningRate = opt.learningRate,
   -- learningRateDecay = 1e-4,
   weightDecay = 0,
   momentum = 0.5
}


if arg[1] then
	dir_name = arg[1]
else
	dir_name = os.date('%B_')..os.date('%D'):sub(4,5)..'_e'..opt.epochs..'_b'..opt.batch_size..'_tr'..opt.train_size..'_tst'..opt.test_size..'_w'..opt.w
	os.execute('mkdir -p '..dir_name)
end

function map(func, array)
	local new_array = {}
	for i,v in ipairs(array) do
		new_array[i] = func(v)
	end
	return new_array
end

function plot(params, fname, xlabel, ylabel, title)
	gnuplot.pngfigure(fname)
	gnuplot.plot(params)
	gnuplot.xlabel(xlabel)
	gnuplot.ylabel(ylabel)
	gnuplot.title(title)
	gnuplot.plotflush()
end

function load_dataset(train_or_test, count)
	-- load
	local data
	if train_or_test == 'train' then
		data = mnist.loadTrainSet(count, {32, 32})
	else
		data = mnist.loadTestSet(count, {32, 32})
	end

	-- vectorize each 2D data point into 1D
	data.data = data.data:reshape(data.data:size(1), 32*32)
	data.data = data.data/255

	print('--------------------------------')
	print(' loaded dataset "' .. train_or_test .. '"')
	print('inputs', data.data:size())
	print('targets', data.labels:size())
	print('--------------------------------')	
	return data.data, data.labels
end

function test(ds, encoder, decoder, criterion, iter)
	local t = encoder:forward(ds)
	t = decoder:forward(t)
	local loss = {}
	for i = 1, ds:size()[1] do
		loss[#loss + 1] = criterion:forward(t[i], ds[i])
	end
	loss = - 1 * torch.DoubleTensor(loss)
	local best10, indices = loss:topk(100)
	local idxFile = string.format(dir_name.."/expt_top100Indices_epoch%d.dat", iter)
	local labels = {}
	torch.save(idxFile, indices)
	for i = 1, 100 do 
		labels[i] = testLabels[indices[i]]
		local x = t[indices[i]]
		x = x:reshape(32,32)
		local fileName = string.format(dir_name.."/expt_l%d_top%d_epoch%d.jpeg", labels[i], i, iter)
		image.save(fileName, x)
	end
	local labelFile = string.format(dir_name.."/expt_top10Labels_epoch%d.dat", iter)
	torch.save(labelFile, torch.ByteTensor(labels))
	return t, -1 * loss
end

function trainOneLayer(opt, ds, ans, model, conjugateModel, criterion, chance)
	train_losses = {}
	for i = 1, opt.epochs do
		print('----- EPOCH ', i, '-----')
		local shuffled_indices =  torch.randperm(opt.train_size, 'torch.LongTensor')
		ds = ds:index(1, shuffled_indices):squeeze()
		ans = ans:index(1, shuffled_indices):squeeze()
		local cur = 1
		local j = 0
		while cur < opt.train_size do
			local cur_ds = ds[{{cur, math.min(cur+opt.batch_size, opt.train_size)},{}}]
			local cur_ans = ans[{{cur, math.min(cur+opt.batch_size, opt.train_size)},{}}]
			cur = cur + opt.batch_size
			j = j + 1
			

			x = {}
			dl_dx = {}
			if chance == 0 then
				x, dl_dx = model:getParameters()
			else
				x, dl_dx = conjugateModel:getParameters()
			end
			

			feval = function(x_new)
			   -- set x to x_new, if differnt
			   -- (in this simple example, x_new will typically always point to x,
			   -- so the copy is really useless)
			   if x ~= x_new then
			      x:copy(x_new)
			   end


			   -- reset gradients (gradients are always accumulated, to accommodate 
			   -- batch methods)
			   dl_dx:zero()

			   -- evaluate the loss function and its derivative wrt x, for that sample
			   
				local outputs = model:forward(cur_ds)
				local outputs_conj = conjugateModel:forward(outputs)
				local loss = criterion:forward(outputs_conj, cur_ans)
				
				local dloss_doutput = criterion:backward(outputs_conj, cur_ans)
				local gradInputConj = conjugateModel:backward(outputs, dloss_doutput)
				local grad = model:backward(cur_ans, gradInputConj)

				if chance == 0 then
					model.gradInput = grad/torch.sqrt(grad:norm(2))
				else
					conjugateModel.gradInput = gradInputConj/torch.sqrt(gradInputConj:norm(2))
				end

				if j % opt.print_every == 0 then
				   print(string.format("Iteration %d, loss %1.6f", j, loss))
				   -- print(dl_dx)
			-- print(loss)
			-- print(x)
				end
				
			    
				return loss, dl_dx
			end


			curr_loss = optim.sgd(feval, x, sgd_params)
			-- model.gradInput = grad/torch.sqrt(grad:norm(2))
			
		end
	end
	return model, conjugateModel
end

function alternateMin(opt, encoder, decoder, criterion, trainDs, testDs)
	local encoder_train_loss = {}
	local decoder_train_loss = {}
	local test_losses = {}
	local iter = 1
	local encoder = encoder
	local decoder = decoder
	-- signal.signal(signal.SIGINT,function(signum)
	-- 								print("Ctrl-C interrupt in alternateMin! Exiting...")
	-- 								enc = encoder
	-- 								dec = decoder
	-- 								enc_tr_loss = encoder_train_loss
	-- 								dec_tr_loss = decoder_train_loss
	-- 								test_loss = test_losses
	-- 								saveAll()
	-- 								os.exit(128 + signum)
	-- 							end)
	while iter < 20 do --Figure out a stopping condition
		print("----- Encoder -----")
		-- local loss = {}
		encoder, decoder = trainOneLayer(opt, trainDs, trainDs:clone(), encoder, decoder, criterion,0)
		-- encoder_train_loss[#encoder_train_loss + 1] = loss
		
		--Simple stopping criterion. Loss smaller than some small number. Can be more sophisticated!
                -- local tmp = encoder_train_loss[#encoder_train_loss]
                -- if tmp[#tmp] < 0.01 then
                --         break
                -- end

		print("----- Decoder -----")
		encoder, decoder = trainOneLayer(opt, trainDs, trainDs:clone(), encoder, decoder, criterion, 1)
		-- decoder_train_loss[#decoder_train_loss + 1] = loss

		--Test
		local t, test_loss = test(testDs, encoder, decoder, criterion, iter)
		test_losses[#test_losses + 1] = test_loss:sum()

		print(string.format("Epoch %4d, test loss = %1.6f", iter, torch.mean(test_loss)))
		iter = iter + 1		
	end
	return encoder, decoder, test_losses
end

function saveAll()
	print('Saving everything...')
	torch.save(dir_name..'/encoder_weights.dat', enc.modules[1].weight)
	torch.save(dir_name..'/decoder_weights.dat', dec.modules[1].weight)
	
	--Save old stuff as well!
	if arg[1] then
		local dn = dir_name..'/old_'..os.date('%B_')..os.date('%D'):sub(4,5)..'_'..os.date('%H%M')
		local etr = torch.load(dir_name..'/enc_loss.dat')
		local dtr = torch.load(dir_name..'/dec_loss.dat')
		local tl  = torch.load(dir_name..'/test_loss.dat')
		for i=1, #enc_tr_loss do
			etr[#etr + 1] = enc_tr_loss[i]
		end
		enc_tr_loss = etr
		for i=1, #dec_tr_loss do
			dtr[#dtr + 1] = dec_tr_loss[i]
		end
		dec_tr_loss = dtr
		for i=1, #tl do
			tl[#tl + 1] = test_loss[i]
		end
		test_loss = tl
		os.execute('mv '..dir_name..'/enc_loss.dat'..' '..dn)
		os.execute('mv '..dir_name..'/dec_loss.dat'..' '..dn)
		os.execute('mv '..dir_name..'/test_loss.dat'..' '..dn)
	end

	torch.save(dir_name..'/enc_loss.dat', enc_tr_loss)
	torch.save(dir_name..'/dec_loss.dat', dec_tr_loss)
	torch.save(dir_name..'/test_loss.dat', test_loss)

	if #enc_tr_loss < 1 or #dec_tr_loss < 1 or #test_loss < 1 then
		os.exit(-1)
	end

	mean_enc_loss = {'Mean Epochal Encoder Loss',
		torch.range(1, #enc_tr_loss),
		torch.Tensor(map(torch.mean, enc_tr_loss)),           
		'-'}
	min_enc_loss = {'Minimum Epochal Encoder Loss',
		torch.range(1, #enc_tr_loss),
		torch.Tensor(map(torch.min, enc_tr_loss)),           
		'-'}
	max_enc_loss = {'Maximum Epochal Encoder Loss',
		torch.range(1, #enc_tr_loss),
		torch.Tensor(map(torch.max, enc_tr_loss)),           
		'-'}


	mean_dec_loss = {'Mean Epochal Decoder Loss',
		torch.range(1, #dec_tr_loss),
		torch.Tensor(map(torch.mean, dec_tr_loss)),          
		'-'}
	min_dec_loss = {'Minimum Epochal Decoder Loss',
		torch.range(1, #dec_tr_loss),
		torch.Tensor(map(torch.min, dec_tr_loss)),           
		'-'}
	max_dec_loss = {'Maximum Epochal Decoder Loss',
		torch.range(1, #dec_tr_loss),
		torch.Tensor(map(torch.max, dec_tr_loss)),          
		'-'}

	tst_loss = {'Test Loss',
		torch.range(1, #test_loss),
		torch.Tensor(test_loss),
		'-'}

	--Plot 
	plot(mean_enc_loss, dir_name..'/Mean_Encoder_Loss.png', 'Epochs', 'Loss', 'Encoder Training Mean Loss Plot')
	plot(min_enc_loss, dir_name..'/Min_Encoder_Loss.png', 'Epochs', 'Loss', 'Encoder Training Minimum Loss Plot')
	plot(max_enc_loss, dir_name..'/Max_Encoder_Loss.png', 'Epochs', 'Loss', 'Encoder Training Maximum Loss Plot')

	plot(mean_dec_loss, dir_name..'/Mean_Decoder_Loss.png', 'Epochs', 'Loss', 'Decoder Training Mean Loss Plot')
	plot(min_dec_loss, dir_name..'/Min_Decoder_Loss.png', 'Epochs', 'Loss', 'Decoder Training Minimum Loss Plot')
	plot(max_dec_loss, dir_name..'/Max_Decoder_Loss.png', 'Epochs', 'Loss', 'Decoder Training Maximum Loss Plot')

	plot(tst_loss, dir_name..'/Test_Loss.png', 'Epochs', 'Loss', 'Test Loss Plot')
end

-- params
inputSize = 32*32
outputSize = 10

-- encoder
encoder = nn.Sequential()
encoder:add(nn.Linear(inputSize,outputSize))
if arg[1] then
	wts = torch.load(dir_name..'/encoder_weights.dat')
	encoder.modules[1].weights = wts
end
encoder:add(nn.Sigmoid())

-- decoder
decoder = nn.Sequential()
decoder:add(nn.Linear(outputSize,inputSize))
if arg[1] then
	wts = torch.load(dir_name..'/decoder_weights.dat')
	decoder.modules[1].weights = wts
end
decoder:add(nn.Sigmoid())

crit = nn.MSECriterion()

--Save old data
if arg[1] then
	dn = dir_name..'/old_'..os.date('%B_')..os.date('%D'):sub(4,5)..'_'..os.date('%H%M')
	os.execute('mkdir -p '..dn)
	os.execute('mv '..dir_name..'/*.png '..dn..'/')
	os.execute('mv '..dir_name..'/*.jpeg '..dn..'/')
	os.execute('mv '..dir_name..'/encoder_weights.dat '..dn..'/')
	os.execute('mv '..dir_name..'/decoder_weights.dat '..dn..'/')
end

trainData = load_dataset('train', opt.train_size)
testData, testLabels = load_dataset('test', opt.test_size)
enc = nil
dec = nil 
enc_tr_loss = nil 
dec_tr_loss = nil 
test_loss = nil
-- print(encoder.modules[1].weights)
-- print(decoder.modules[1].weights)
enc, dec, test_loss= alternateMin(opt, encoder, decoder, crit, trainData, testData)
-- local t, test_loss = test(testData, encoder, decoder, crit, 0)
-- print(string.format("Epoch %4d, test loss = %1.6f", iter, torch.mean(test_loss)))


saveAll()


