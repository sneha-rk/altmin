require 'torch'
require 'dataset-mnist'

require 'nn'
require 'image'
require 'os'
require 'gnuplot'
require 'paths'
signal = require('posix.signal')

dataset = 'mnist'
activation = 'relu'


opt = 
{
	n_epochs = 200,
	epochs = 1,
	batch_size = 500,

	print_every = 10,  
	learningRate = 5.0e-3,
	epsilon = 1e-4,

	inputSize = 32*32,
	outputSize = 500,	
	sizes = {1024, 750, 500}, --Sizes of inputs in various layers.
	
	k = 2, --Number of hidden layers
	train_size = 60000,
	channels = 1,
	test_size = 10000
}	

opt.k = #opt.sizes 

if arg[1] then
	dir_name = arg[1]
else
	dir_name = 'c'..dataset..os.date('%B_')..os.date('%D'):sub(4,5)..os.date('%X'):sub(4,5)..'_e'..opt.epochs..'_b'..opt.batch_size..'_tr'..opt.train_size..'_tst'..opt.test_size
	os.execute('mkdir -p '..dir_name)
end

-- Log architecture!
log_f = io.open("arch_logging.csv", "a+")
log_f:write(string.format("%s, %1.6f, %d", dir_name, opt.learningRate, opt.k))
for i = 1, opt.k do 
	log_f:write(", %d", sizes[i])
end
log_f:write("\n")
log_f:close()

test_f = io.open(dir_name.. "/test_error.csv", "a+")
train_f = io.open(dir_name.."/train_error.csv", "a+")



-- convert rgb to grayscale by averaging channel intensities
function rgb2gray(im)
	-- Image.rgb2y uses a different weight mixture

	local dim, w, h = im:size()[1], im:size()[2], im:size()[3]
	if dim ~= 3 then
		 print('<error> expected 3 channels')
		 return im
	end

	-- a cool application of tensor:select
	local r = im:select(1, 1)
	local g = im:select(1, 2)
	local b = im:select(1, 3)

	local z = torch.Tensor(w, h):zero()

	-- z = z + 0.21r
	z = z:add(0.21, r)
	z = z:add(0.72, g)
	z = z:add(0.07, b)
	return z
end


-- Loading appropriate dataset
if dataset == 'cifar' then
	print('Using CIFAR-10 Dataset...')
	require 'dataset-cifar'

	opt.train_size = 50000
	opt.channels = 3
elseif dataset == 'gcifar' then
	print('Using CIFAR-10 Dataset...')
	require 'dataset-cifar'

	opt.train_size = 50000
	opt.channels = 1
else	
	opt.train_size = 60000
	opt.channels = 1
end
opt.sizes[1] = opt.channels * 1024
-- opt.sizes[5] = opt.channels * 1024

-- opt.sizes = {1024, 500,1024}
-- opt.k = 1


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
	if dataset == 'gcifar' then 
		data.data =data.data
	else
		data.data = data.data/255.0
	end

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
elseif dataset == 'gcifar' then 
	trainData = torch.Tensor(opt.train_size, 32*32) 
	testData = torch.Tensor(opt.test_size, 32*32)
	testLabels = cifar.testData.labels

	for i=1,opt.train_size do
		trainData[i] = image.rgb2y(cifar.trainData.data[i]) 
	end

	for i=1,opt.test_size do
		testData[i] = image.rgb2y(cifar.testData.data[i]) 
	end

	-- image.save('example.jpg', trainData[1])
	-- trainData = image.rgb2y(cifar.trainData.data) / 255.0
	-- testData, testLabels = image.rgb2y(cifar.testData.data) / 255.0, cifar.testData.labels
else	
	trainData = load_dataset('train', opt.train_size)
	testData, testLabels = load_dataset('test', opt.test_size)
end

function test(ds, encoder, decoder, criterion, iter, flag)
	local t = encoder:forward(ds)
	t = decoder:forward(t)
	local loss = {}
	for i = 1, ds:size()[1] do
		loss[#loss + 1] = criterion:forward(t[i], ds[i])
	end
	loss = - 1 * torch.DoubleTensor(loss)
	if flag == 1 then
		local best10, indices = loss:topk(100)
		local idxFile = string.format(dir_name.."/expt_top100Indices_epoch%d.dat", iter)
		local labels = {}
		torch.save(idxFile, indices)
		for i = 1, 100 do 
			labels[i] = testLabels[indices[i]]
			local x = t[indices[i]]
			local orig = ds[indices[i]]
			if dataset == 'cifar' then
				x = x:reshape(3, 32,32)
				orig = orig:reshape(3, 32,32)
			else 
				x = x:reshape(32,32)
				orig = orig:reshape(32,32)
			end 

			local fileName = string.format(dir_name.."/expt_l%d_top%d_epoch%d.jpeg", labels[i] + 1, i, iter)
			image.save(fileName, x)
			local ofileName = string.format(dir_name.."/expt_l%d_top%d_epoch%dorig.jpeg", labels[i] + 1, i, iter)
			image.save(ofileName, orig)
		end
		local labelFile = string.format(dir_name.."/expt_top10Labels_epoch%d.dat", iter)
		torch.save(labelFile, torch.ByteTensor(labels))
	end
	return t, -1 * loss
end

function trainOneLayer(opt, ds, model, conjugateModel, criterion, testDs, iter)
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

			if j % opt.print_every == 0 then
			   local t, test_loss = test(testDs, model, conjugateModel, criterion, (iter * opt.train_size / opt.batch_size) + j, 0)
			   print(string.format("%d, %1.6f", (iter * opt.train_size / opt.batch_size) + j, torch.mean(test_loss)))

			   test_f:write(string.format("%d, %1.6f\n", ((iter * opt.k ) * opt.train_size / opt.batch_size) + j, torch.mean(test_loss)))
			   train_f:write(string.format("%d, %1.6f\n", ((iter * opt.k ) * opt.train_size / opt.batch_size) + j, loss))
			end
			
			train_losses[#train_losses + 1] = loss -- append the new loss
			j = j + 1
		end
	end
	return model, conjugateModel, train_losses
end

function train(opt, encoder, decoder, criterion, trainDs, testDs)
	local encoder_train_error = {}
	local decoder_train_error = {}
	local iter = 0

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

	while iter < opt.n_epochs do --Figure out a stopping condition
		-- print("----- Encoder -----")
		encoder, decoder, tmp = trainOneLayer(opt, trainDs, encoder, decoder, criterion, testDs, iter)
		encoder_train_error[#encoder_train_error + 1] = tmp

		iter = iter + 1
		local t, err = test(testDs, encoder, decoder, criterion, iter, 1)
	end
	return encoder, decoder, encoder_train_error, decoder_train_error
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

-- verbose
print('==> Constructed linear auto-encoder')

-- Training the autoencoder.

enc, dec, enc_tr, dec_tr = train(opt, encoder, decoder, crit, trainData, testData)

saveAll()


