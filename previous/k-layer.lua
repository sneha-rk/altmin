require 'torch'
require 'dataset-mnist'
require 'nn'
require 'image'
require 'os'
require 'gnuplot'
signal = require('posix.signal')

dataset = ''
activation = 'relu'


opt = 
{
	n_epochs = 200,
	epochs = 1,
	batch_size = 500,


	inputSize = 32*32,
	outputSize = 500,
	print_every = 10,  
	learningRate = 1e-3,
	epsilon = 1e-4,
	
	sizes = {}, --Sizes of inputs in various layers.
	k = 1, --Number of hidden layers
	train_size = 50000,
	test_size = 10000,
	channels = 3
}	
if not (assert(#opt.sizes == opt.k + 2, "opt.sizes must have size equal to opt.k + 2")) then
	os.exit(-1)	
end
print(opt)

if arg[1] then
	dir_name = arg[1]
else
	dir_name = os.date('%B_')..os.date('%D'):sub(4,5)..'_e'..opt.epochs..'_b'..opt.batch_size..'_tr'..opt.train_size..'_tst'..opt.test_size
	os.execute('mkdir -p '..dir_name)
end

-- Loading appropriate dataset
if dataset == 'cifar' then
	print('Using CIFAR-10 Dataset...')
	require 'dataset-cifar'

	opt.train_size = 50000
	opt.channels = 3
	opt.sizes = {opt.inputSize * opt.channels, opt.outputSize, opt.inputSize * opt.channels}
else	
	opt.train_size = 60000
	opt.channels = 1
	opt.sizes = {opt.inputSize * opt.channels, opt.outputSize, opt.inputSize * opt.channels}
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

function test(ds, model, criterion, iter, flag)
	local t = ds
	for i = 1, opt.k + 1 do
		t = model[i]:forward(t)
	end
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

function trainOneLayer(opt, testDs, ds, ans, model, criterion, k, iter)
	train_losses = {}
	for i = 1, opt.epochs do
		-- print('----- EPOCH ', i, '-----')
		local shuffled_indices =  torch.randperm(opt.train_size, 'torch.LongTensor')
		ds = ds:index(1, shuffled_indices):squeeze()
		ans = ans:index(1, shuffled_indices):squeeze()
		local cur = 1
		local j = 0
		while cur < opt.train_size do
			local cur_ds = ds[{{cur, math.min(cur+opt.batch_size, opt.train_size)},{}}]
			local cur_ans = ans[{{cur, math.min(cur+opt.batch_size, opt.train_size)},{}}]
			cur = cur + opt.batch_size
			
			--outputs[i+1] is the output of layer i
			local outputs = {cur_ds}
			for i = 1, opt.k + 1 do
				outputs[i+1] = model[i]:forward(outputs[i])
			end

			local output = outputs[opt.k + 2]
			local loss = criterion:forward(output, cur_ans)

			
			local grad = criterion:backward(output, cur_ans)
			for i = opt.k + 1, 1, -1 do
				grad = model[i]:backward(outputs[i], grad) 
			end

			model[k].gradInput = grad/torch.sqrt(grad:norm(2))
			model[k]:updateParameters(opt.learningRate)

			if j % opt.print_every == 0 then
			   local t, test_loss = test(testDs, model, criterion, ((iter * opt.k + k) * opt.train_size / opt.batch_size) + j, 0)
			   print(string.format("%d, %1.6f", ((iter * opt.k + k) * opt.train_size / opt.batch_size) + j, torch.mean(test_loss)))
			end

			train_losses[#train_losses + 1] = loss -- append the new loss
			j = j + 1
		end
	end
	return model, torch.DoubleTensor(train_losses)
end

function alternateMin(opt, model, criterion, trainDs, testDs)
	local train_losses = {}
	for i = 1, opt.k + 1 do
		table.insert(train_losses, {})
	end
	local test_losses = {}
	local iter = 0
	while iter < opt.n_epochs do --Figure out a stopping condition
		for i = 1, opt.k + 1 do
			local loss = {}
			model, loss = trainOneLayer(opt,testDs, trainDs, trainDs:clone(), model, criterion, i, iter)
			train_losses[i][#train_losses[i] + 1] = loss		
			--Simple stopping criterion. Loss smaller than some small number. Can be more sophisticated!
            local tmp = train_losses[i][#train_losses[i]]
            if tmp[#tmp] < 0.01 then
                    break
            end
        end
		--Test
		local t, test_loss = test(testDs, model, criterion, iter, 1)
		test_losses[#test_losses + 1] = test_loss:sum()

		iter = iter + 1		
	end
	return model, train_losses, test_losses
end

function saveAll(model, train_losses, test_losses)
	print('Saving everything...')
	for i = 1, opt.k + 1 do
		torch.save(dir_name..'/weights_'..i..'.dat', model[i].modules[1].weight)
	end
	torch.save(dir_name..'/test_loss.dat', test_losses)
	torch.save(dir_name..'/train_loss.dat', train_losses)

	for i = 1, opt.k do
		mean_loss = {'Mean Epochal Loss, Layer '..i,
			torch.range(1, #train_losses[i]), 
			torch.Tensor(map(torch.mean, train_losses[i])),
			'-'}
		min_loss = {'Minimum Epochal Loss, Layer '..i,
			torch.range(1, #train_losses[i]), 
			torch.Tensor(map(torch.min, train_losses[i])),
			'-'}
		max_loss = {'Maximum Epochal Loss, Layer '..i,
			torch.range(1, #train_losses[i]), 
			torch.Tensor(map(torch.max, train_losses[i])),
			'-'}
		plot(mean_loss, dir_name..'/Mean_loss_layer_'..i..'.png', 'Epochs', 'Mean Loss', 'Layer '..i..'Mean Loss Plot')
		plot(min_loss, dir_name..'/Minimum_loss_layer_'..i..'.png', 'Epochs', 'Minimum Loss', 'Layer '..i..'Minimum Loss Plot')
		plot(max_loss, dir_name..'/Maximum_loss_layer_'..i..'.png', 'Epochs', 'Maximum Loss', 'Layer '..i..'Maximum Loss Plot')
	end

	tst_loss = {'Test Loss',
		torch.range(1, #test_losses),
		torch.Tensor(test_losses),
		'-'}

	plot(test_loss, dir_name..'/Test_Loss.png', 'Epochs', 'Loss', 'Test Loss Plot')
end

model = {}
-- encoder
for i = 1, opt.k + 1 do
	layer = nn.Sequential()
	layer:add(nn.Linear(opt.sizes[i], opt.sizes[i+1]))
	if activation == 'relu' then
		layer:add(nn.LeakyReLU())
	else 
		layer:add(nn.Sigmoid())
	end
	table.insert(model, layer)
	print(layer)
end

crit = nn.MSECriterion()



m = nil
tr_loss = nil
tst_loss = nil
m, tr_loss, tst_loss= alternateMin(opt, model, crit, trainData, testData)
saveAll(m, tr_loss, tst_loss)


