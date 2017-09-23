require 'torch'
require 'optim'
require 'dataset-mnist'
require 'nn'
require 'image' 
require 'pl'
require 'paths'

opt = 
{
	epochs = 1,
	batch_size = 50,
	print_every = 250,  
	train_size = 60000,
	test_size = 10000,
	epsilon = 1e-1,
	w = 1.6, --Recheck. This slows stuff down!
	learningRate = 1e-1, --ToDo: Set Learning Rate
}

classes = {'1','2','3','4','5','6','7','8','9','10'}

opt.learningRate = opt.epsilon * torch.exp(-opt.w)
confusion = optim.ConfusionMatrix(classes)

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
	
	return data.data, data.labels
end

-- test function
function test(testDs, testLabel, encoder, decoder, criterion, iter)
   -- local vars
   local time = sys.clock()

   -- test over given dataset
   print('<trainer> on testing Set:')
   for t = 1,opt.test_size,opt.batch_size do
   	  
   	  local inputs = testDs[{{t, math.min(t+opt.batch_size, opt.test_size)},{}}]
	  local targets = testLabel[{{t, math.min(t+opt.batch_size, opt.test_size)}}]

      -- -- create mini batch
      -- local inputs = torch.Tensor(opt.batch_size,1,32,32)
      -- local targets = torch.Tensor(opt.batch_size)
      -- local k = 1
      -- for i = t,math.min(t+opt.batch_size-1,dataset:size()) do
      --    -- load new sample
      --    local sample = dataset[i]
      --    local input = sample[1]:clone()
      --    local _,target = sample[2]:clone():max(1)
      --    target = target:squeeze()
      --    inputs[k] = input
      --    targets[k] = target
      --    k = k + 1
      -- end

      -- test samples
      local p1 = encoder:forward(inputs)
      local preds = decoder:forward(p1)
      -- local preds = criterion:forward(p2)

      -- confusion:
      for i = 1,opt.batch_size do
         confusion:add(preds[i], targets[i])
      end
   end

   -- timing
   time = sys.clock() - time
   time = time / opt.test_size
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   print('Mean class accuracy (test set)' .. (confusion.totalValid * 100))
   confusion:zero()
end


-- function test(ds, testLabel, encoder, decoder, criterion, iter)
-- 	local t = encoder:forward(ds)
-- 	t = decoder:forward(t)
-- 	local err = {}
-- 	for i = 1, ds:size()[1] do
-- 		err[#err + 1] = criterion:forward(t[i], testLabel[i])
-- 	end
-- 	-- err = - 1 * torch.DoubleTensor(err)
-- 	print(err)

-- 	-- local best10, indices = err:topk(10)
-- 	-- for i = 1, 10 do 
-- 	-- 	local x = t[indices[i]] 
-- 	-- 	x = x:reshape(32,32)
-- 	-- 	local fileName = string.format("control_top%d_iter%d.jpeg", i, iter)
-- 	-- 	-- print(x)
-- 	-- 	-- print(x:round())
-- 	-- 	image.save(fileName, x)
-- 	-- end


-- 	return t, err
-- end

function trainOneLayer(opt, ds, trainLabel, model, conjugateModel, criterion)
	train_losses = {}
	for i = 1, opt.epochs do

		print('----- EPOCH ', i, '-----')
		local shuffled_indices =  torch.randperm(opt.train_size, 'torch.LongTensor')
		ds = ds:index(1, shuffled_indices):squeeze()
		trainLabel = trainLabel:index(1, shuffled_indices):squeeze()
		-- print(ds)
		local cur = 1
		local j = 0
		while cur < opt.train_size do
			local cur_ds = ds[{{cur, math.min(cur+opt.batch_size, opt.train_size)},{}}]
			local cur_label = trainLabel[{{cur, math.min(cur+opt.batch_size, opt.train_size)}}]
			cur = cur + opt.batch_size
		 	j = j + 1
			
			--get parameters
			local mParams, mGrads = model:getParameters()
			mGrads:zero()

			-- Forward
			local outputs = model:forward(cur_ds)
			local outputs_conj = conjugateModel:forward(outputs)
			local loss = criterion:forward(outputs_conj, cur_label)

			-- Backward
			local dloss_doutput = criterion:backward(outputs_conj, cur_label)
			local gradInputConj = conjugateModel:backward(outputs, dloss_doutput)
			local grad = model:backward(cur_ds, gradInputConj)

			-- Update weights.
			model.gradInput = grad/torch.sqrt(grad:norm(2))
			model:updateParameters(opt.learningRate)

			conjugateModel.gradInput = gradInputConj/torch.sqrt(gradInputConj:norm(2))
			conjugateModel:updateParameters(opt.learningRate)
			
			-- train_losses[#train_losses + 1] = loss -- append the new loss
		end
	end
	return model, conjugateModel, train_losses
end

function train(opt, encoder, decoder, criterion, trainDs, trainLabel, testDs, testLabel)
	local encoder_train_error = {}
	local decoder_train_error = {}
	local iter = 1
	while true do --Figure out a stopping condition
		print("----- Encoder -----")
		encoder, decoder, tmp = trainOneLayer(opt, trainDs, trainLabel, encoder, decoder, criterion)
		encoder_train_error[#encoder_train_error + 1] = tmp

		--Test
		test(testDs, testLabel, encoder, decoder, criterion, iter)

		-- print(string.format("iteration %4d, test error = %1.6f", iter, -1 * torch.mean(err)))
		iter = iter + 1

		if iter > 30 then
			break
		end
	end
	return encoder, decoder, encoder_train_error, decoder_train_error
end

-- Load data
trainData, trainLabel = load_dataset('train', opt.train_size)
testData, testLabel = load_dataset('test', opt.test_size)

-- params
inputSize = 32*32
layerSize = 2000
outputSize = 10

encoder = nn.Sequential()
encoder:add(nn.Linear(inputSize, layerSize))
encoder:add(nn.Sigmoid())

-- decoder
decoder = nn.Sequential()
decoder:add(nn.Linear(layerSize, #classes))
decoder:add(nn.LogSoftMax())
crit = nn.ClassNLLCriterion()

-- verbose
print('==> Constructed linear auto-encoder')

-- Training the autoencoder.

enc, dec, enc_tr, dec_tr = train(opt, encoder, decoder, crit, trainData, trainLabel, testData, testLabel)

torch.save('encoder.dat', enc)
torch.save('decoder.dat', dec)
torch.save('en_err.dat', enc_tr)
torch.save('dec_err.dat', dec_tr)

