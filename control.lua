require 'torch'
-- require 'unsup'
require 'dataset-mnist'
require 'nn'
require 'image'

opt = 
{
	epochs = 2,
	batch_size = 500,
	print_every = 250,  
	train_size = 60000,
	test_size = 1000,
	epsilon = 1e-4,
	w = 0, --Recheck. This slows stuff down!
	learningRate = 1e-1, --ToDo: Set Learning Rate
}

opt.learningRate = opt.epsilon * torch.exp(-opt.w)

if arg[1] then
	dir_name = arg[1]
else
	dir_name = 'c'..os.date('%B_')..os.date('%D'):sub(4,5)..'_e'..opt.epochs..'_b'..opt.batch_size..'_tr'..opt.train_size..'_tst'..opt.test_size..'_w'..opt.w
	os.execute('mkdir -p '..dir_name)
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
	data.data = data.data / 255

	print('--------------------------------')
	print(' loaded dataset "' .. train_or_test .. '"')
	print('inputs', data.data:size())
	print('targets', data.labels:size())
	print('--------------------------------')
	
	return data.data
end

function test(ds, encoder, decoder, criterion, iter)
	local t = encoder:forward(ds)
	t = decoder:forward(t)
	local err = {}
	for i = 1, ds:size()[1] do
		err[#err + 1] = criterion:forward(t[i], ds[i])
	end
	err = - 1 * torch.DoubleTensor(err)
	local best10, indices = err:topk(100)
	for i = 1, 100 do 
		local x = t[indices[i]] 
		x = x:reshape(32,32)
		local fileName = string.format(dir_name.."/control_top%d_iter%d.jpeg", i, iter)
		-- print(x)
		-- print(x:round())
		image.save(fileName, x)
	end

	return t, err
end

function trainOneLayer(opt, ds, model, conjugateModel, criterion)
	train_losses = {}
	for i = 1, opt.epochs do

		-- print('----- EPOCH ', i, '-----')
		local shuffled_indices =  torch.randperm(opt.train_size, 'torch.LongTensor')
		ds = ds:index(1, shuffled_indices):squeeze()
		local cur = 1
		local j = 0
		while cur < opt.train_size do
			local cur_ds = ds[{{cur, math.min(cur+opt.batch_size, opt.train_size)},{}}]
			cur = cur + opt.batch_size
		 	j = j + 1
			
			--get parameters
			local mParams, mGrads = model:getParameters()
			mGrads:zero()

			-- Forward
			local outputs = model:forward(cur_ds)
			local outputs_conj = conjugateModel:forward(outputs)
			local loss = criterion:forward(outputs_conj, cur_ds)

			-- Backward
			local dloss_doutput = criterion:backward(outputs_conj, cur_ds)
			local gradInputConj = conjugateModel:backward(outputs, dloss_doutput)
			local grad = model:backward(cur_ds, gradInputConj)

			-- Update weights
			model:updateParameters(opt.learningRate)
			conjugateModel:updateParameters(opt.learningRate)
			
			train_losses[#train_losses + 1] = loss -- append the new loss
		end
	end
	return model, conjugateModel, train_losses
end

function train(opt, encoder, decoder, criterion, trainDs, testDs)
	local encoder_train_error = {}
	local decoder_train_error = {}
	local iter = 1
	while true do --Figure out a stopping condition
		-- print("----- Encoder -----")
		encoder, decoder, tmp = trainOneLayer(opt, trainDs, encoder, decoder, criterion)
		encoder_train_error[#encoder_train_error + 1] = tmp

		--Test
		local t, err = test(testDs, encoder, decoder, criterion, iter)

		print(string.format("%d, %1.6f", iter, -1 * torch.mean(err)))
		-- print(string.format("iteration %4d, test error = %1.6f", iter, -1 * torch.mean(err)))
		iter = iter + 1
		
		--Simple stopping criterion. Loss smaller than some small number. Can be more sophisticated!
		if  -1 * torch.mean(err) < 0.01 then
			break
		end
		-- if iter > 500 then
		-- 	break
		-- end
	end
	return encoder, decoder, encoder_train_error, decoder_train_error
end


function saveAll()
	print('Saving everything...')
	torch.save(dir_name..'/encoder_weights.dat', enc.modules[1].weight)
	torch.save(dir_name..'/decoder_weights.dat', dec.modules[1].weight)

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

-- Load data
trainData = load_dataset('train', opt.train_size)
testData = load_dataset('test', opt.test_size)

-- params
inputSize = 32*32
outputSize = 100

-- encoder
encoder = nn.Sequential()
encoder:add(nn.Linear(inputSize,outputSize))
encoder:add(nn.Sigmoid())

-- decoder
decoder = nn.Sequential()
decoder:add(nn.Linear(outputSize,inputSize))
decoder:add(nn.Sigmoid())

crit = nn.MSECriterion()

-- verbose
print('==> Constructed linear auto-encoder')

-- Training the autoencoder.

enc, dec, enc_tr, dec_tr = train(opt, encoder, decoder, crit, trainData, testData)

saveAll()
-- torch.save('encoder.dat', enc)
-- torch.save('decoder.dat', dec)
-- torch.save('en_err.dat', enc_tr)
-- torch.save('dec_err.dat', dec_tr)