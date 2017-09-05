require 'torch'
require 'dataset-mnist'
require 'nn'
require 'image'
require 'os'
require 'gnuplot'
require 'io'
signal = require('posix.signal')

dataset = 'mnist'
activation = 'relu'

opt = 
{
	epochs = 1,
	batch_size = 500,
	print_every = 10,  
	train_size = 60000,
	test_size = 10000,
	epsilon = 1.0e-4,
	sizes = {1024, 2000, 1000}, --Sizes of inputs in various layers.
	learningRate = 1.0e-4, --SET PROPERLY
	channels = 1
}


opt.k = #opt.sizes --Number of hidden layers is 2k - 3

print(opt)

if arg[1] then
	dir_name = arg[1]
else
	dir_name = dataset..os.date('%B_')..os.date('%D'):sub(4,5)..os.date('%X'):sub(4,5)..'_e'..opt.epochs..'_b'..opt.batch_size..'_tr'..opt.train_size..'_tst'..opt.test_size
	os.execute('mkdir -p '..dir_name)
end

-- Log architecture!
log_f = io.open("arch_logging.lua", "a+")
log_f:write(string.format("%s, %1.6f, %d", dir_name, opt.learningRate, opt.k))
-- log_f:write(opt)
log_f:write("\n")
log_f:close()


test_f = io.open(dir_name.. "/test_error.csv", "a+")
train_f = io.open(dir_name.."/train_error.csv", "a+")

-- Loading appropriate dataset
if dataset == 'cifar' then
	print('Using CIFAR-10 Dataset...')
	require 'dataset-cifar'

	opt.train_size = 50000
	opt.channels = 3
else	
	opt.train_size = 60000
	opt.channels = 1
end
opt.sizes[1] = opt.channels * 32 * 32

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
	data.data = data.data/255.0

	print('--------------------------------')
	print(' loaded dataset "' .. train_or_test .. '"')
	print('inputs', data.data:size())
	print('targets', data.labels:size())
	print('--------------------------------')	
	return data.data, data.labels
end

-- Loading appropriate dataset
if dataset == 'cifar' then
	trainData = cifar.trainData.data:reshape(cifar.trainData.data:size(1), 3*32*32) / 255.0
	testData, testLabels = cifar.testData.data:reshape(cifar.testData.data:size(1), 3*32*32) / 255.0, cifar.testData.labels
else	
	trainData = load_dataset('train', opt.train_size)
	testData, testLabels = load_dataset('test', opt.test_size)
end


function test(ds, encoder, decoder, criterion, iter, imgflag)
	local t = encoder:forward(ds)
	t = decoder:forward(t)
	local loss = {}
	for i = 1, ds:size()[1] do
		loss[#loss + 1] = criterion:forward(t[i], ds[i])
	end
	loss = - 1 * torch.DoubleTensor(loss)
	if imgflag == 1 then
		local best10, indices = loss:topk(100)
		local idxFile = string.format(dir_name.."/expt_top100Indices_epoch%d.dat", iter)
		local labels = {}
		torch.save(idxFile, indices)
		for i = 1, 100 do 
			labels[i] = testLabels[indices[i]]
			local x = t[indices[i]]
			if dataset == 'cifar' then
				x = x:reshape(3, 32,32)
			else 
				x = x:reshape(32,32)
			end 

			local fileName = string.format(dir_name.."/expt_l%d_top%d_epoch%d.jpeg", labels[i] + 1, i, iter)
			image.save(fileName, x)
		end
		local labelFile = string.format(dir_name.."/expt_top10Labels_epoch%d.dat", iter)
		torch.save(labelFile, torch.ByteTensor(labels))
	end
	return t, -1 * loss
end


function trainOneLayer(opt, ds, ans, encoder, decoder, criterion, l, iter,testDs)
	train_losses_enc = {}
	train_losses_dec = {}
	for i = 1, opt.epochs do
		-- print('----- EPOCH ', i, '-----')
		--Shuffle dataset
		local shuffled_indices =  torch.randperm(opt.train_size, 'torch.LongTensor')
		ds = ds:index(1, shuffled_indices):squeeze()
		ans = ans:index(1, shuffled_indices):squeeze()
		local cur = 1 -- Start of minibatch
		local j = 0   -- Itercount

		-- current epoch
		while cur < opt.train_size do
			-- Take current minibatch
			local cur_ds = ds[{{cur, math.min(cur+opt.batch_size, opt.train_size)},{}}]
			local cur_ans = ans[{{cur, math.min(cur+opt.batch_size, opt.train_size)},{}}]
			-- This will be next batch start 
			cur = cur + opt.batch_size
			
			-- We need to take current minibatch, fwd update enc, then fwd update dec
			for dec = 0, 1 do
				-- Forward
				local encoder_outputs = encoder:forward(cur_ds)
				local decoder_outputs = decoder:forward(encoder_outputs)
				local loss = criterion:forward(decoder_outputs, cur_ans)

				-- Backward

				local dloss_doutput = criterion:backward(decoder_outputs, cur_ans)
				local encoder_grad_input = decoder:backward(encoder_outputs, dloss_doutput)
				local grad = encoder:backward(cur_ds, encoder_grad_input)
				
				-- Update
				if dec == 0 then
					layer = encoder.modules[2*l - 1]
					_, grad = layer:getParameters()
					if grad:norm(2) < 1e-7 then
						break
					end
					layer.gradInput = grad / grad:norm(2)
					layer:updateParameters(opt.learningRate)
				else
					l = opt.k - l
					layer = decoder.modules[2*l - 1]
					_, grad = layer:getParameters()
					if grad:norm(2) < 1e-7 then
						break
					end
					layer.gradInput = grad / grad:norm(2)
					layer:updateParameters(opt.learningRate)
				end
				
				-- Print loss every opt.print_every minibatch updates
				if j % opt.print_every == 0 then
				   local t, test_loss = test(testDs, encoder, decoder, criterion, ((iter * opt.k + l) * opt.train_size / opt.batch_size) + j, 0)
				   print(string.format("%d, %1.6f", ((iter * opt.k + l) * opt.train_size / opt.batch_size) + j, torch.mean(test_loss)))

				   test_f:write(string.format("%d, %1.6f\n", ((iter * opt.k + l) * opt.train_size / opt.batch_size) + j, torch.mean(test_loss)))
				   train_f:write(string.format("%d, %1.6f\n", ((iter * opt.k + l) * opt.train_size / opt.batch_size) + j, loss))
				end
				j = j + 1

				if dec == 0 then 
					train_losses_enc[#train_losses_enc + 1] = loss -- append the new loss
				else
					train_losses_dec[#train_losses_dec + 1] = loss -- append the new loss
				end

			end

			-- Now onto the next update
		end
	end
	return encoder, decoder, torch.DoubleTensor(train_losses_enc), torch.DoubleTensor(train_losses_dec)
end

function alternateMin(opt, encoder, decoder, criterion, trainDs, testDs)
	local encoder_train_loss = {}
	local decoder_train_loss = {}
	local test_losses = {}
	local iter = 1
	local encoder = encoder
	local decoder = decoder

	-- for i = 1, opt.k do
	-- 	table.insert(encoder_train_loss, {})
	-- 	table.insert(decoder_train_loss, {})
	-- end

	signal.signal(signal.SIGINT,function(signum)
									print("Ctrl-C interrupt in alternateMin! Exiting...")
									enc = encoder
									dec = decoder
									enc_tr_loss = encoder_train_loss
									dec_tr_loss = decoder_train_loss
									test_loss = test_losses
									saveAll()
									os.exit(128 + signum)
								end)
	while true do --Figure out a stopping condition

		local loss = {}

		for l = 1, opt.k - 1 do
			-- print(l)
			--Train Encoder			
			-- print('Encoder')
			encoder, decoder, loss_enc, loss_dec = trainOneLayer(opt, trainDs, trainDs:clone(), encoder, decoder, criterion, l, iter,testDs)
			encoder_train_loss[#encoder_train_loss + 1] = loss_enc
			
			--Train Decoder
			-- print('Decoder')
			-- encoder, decoder, loss = trainOneLayer(opt, trainDs, trainDs:clone(), encoder, decoder, criterion, l, iter,testDs,false)
			decoder_train_loss[#decoder_train_loss + 1] = loss_dec
		end


		--Test
		local t, test_loss = test(testDs, encoder, decoder, criterion, iter, 1)
		test_losses[#test_losses + 1] = test_loss:sum()

		-- print(string.format("Epoch %4d, test loss = %1.6f", iter, torch.mean(test_loss)))
		print(string.format("%d, %1.6f", iter, torch.mean(test_loss)))
		iter = iter + 1		
	end
	return encoder, decoder, encoder_train_loss, decoder_train_loss, test_losses
end

function saveAll()
	print('Saving everything...')
	for i = 1, opt.k - 1 do
		torch.save(dir_name..string.format('/encoder_weights_%d.dat', i), enc.modules[2*i - 1].weight)
		torch.save(dir_name..string.format('/decoder_weights_%d.dat', i), dec.modules[2*i - 1].weight)
	end
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

	test_f:close()
	train_f:close()
end

-- encoder
encoder = nn.Sequential()

for i = 1, opt.k - 1 do
	encoder:add(nn.Linear(opt.sizes[i], opt.sizes[i+1]))
	encoder:add(nn.LeakyReLU())
	if arg[1] then
		wts = torch.load(dir_name..string.format('/encoder_weights_%d.dat', i))
		encoder.modules[2*i - 1].weights = wts
	end
end
print(encoder)

-- decoder
decoder = nn.Sequential()

for i = opt.k , 2, -1 do
	j = opt.k + 1 - i
	decoder:add(nn.Linear(opt.sizes[i], opt.sizes[i-1]))
	decoder:add(nn.LeakyReLU())
	if arg[1] then
		wts = torch.load(dir_name..string.format('/decoder_weights_%d.dat', j))
		decoder.modules[2*j - 1].weights = wts
	end
end
print(decoder)

crit = nn.MSECriterion()

--Save old data
if arg[1] then
	dn = dir_name..'/old_'..os.date('%B_')..os.date('%D'):sub(4,5)..'_'..os.date('%H%M')
	os.execute('mkdir -p '..dn)
	os.execute('mv '..dir_name..'/*.png '..dn..'/')
	os.execute('mv '..dir_name..'/*.jpeg '..dn..'/')
	for i = 1, opt.k do
		os.execute('mv '..dir_name..string.format('/encoder_weights_%d.dat ', i)..dn..'/')
		os.execute('mv '..dir_name..string.format('/decoder_weights_%d.dat ', i)..dn..'/')
	end
end

-- trainData = load_dataset('train', opt.train_size)
-- testData, testLabels = load_dataset('test', opt.test_size)
enc = nil
dec = nil 
enc_tr_loss = nil 
dec_tr_loss = nil 
test_loss = nil

enc, dec, enc_tr_loss, dec_tr_loss, test_loss= alternateMin(opt, encoder, decoder, crit, trainData, testData)

saveAll()