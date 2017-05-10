require 'torch'
require 'dataset-mnist'
require 'nn'
require 'image'
require 'os'
require 'gnuplot'
signal = require('posix.signal')

opt = 
{
	epochs = 2,
	batch_size = 500,
	print_every = 250,  
	train_size = 60000,
	test_size = 1000,
	learningRate = 1e-4,
	epsilon = 1e-4,
	k = 2, --Number of hidden layers
	sizes = {1024, 100, 100, 1024} --Sizes of inputs in various layers.
}	

if not (assert(#opt.sizes == opt.k + 2, "opt.sizes must have size equal to opt.k + 2")) then
	os.exit(-1)	
end

print(opt)

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
	data.data = data.data/255.0

	print('--------------------------------')
	print(' loaded dataset "' .. train_or_test .. '"')
	print('inputs', data.data:size())
	print('targets', data.labels:size())
	print('--------------------------------')	
	return data.data, data.labels
end

function test(ds, model, criterion, iter)
	local t = ds
	for i = 1, opt.k do
		t = model[i]:forward(t)
	end
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
		local fileName = string.format(dir_name.."/expt_l%d_top%d_epoch%d.jpeg", labels[i] + 1, i, iter)
		image.save(fileName, x)
	end
	local labelFile = string.format(dir_name.."/expt_top10Labels_epoch%d.dat", iter)
	torch.save(labelFile, torch.ByteTensor(labels))
	return t, -1 * loss
end

function trainOneLayer(opt, ds, ans, model, criterion, k)
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
			j = j + 1
			
			--outputs[i+1] is the output of layer i
			local outputs = {cur_ds} 
			for i = 1, opt.k do
				outputs[i+1] = model[i]:forward(outputs[i])
			end

			local loss = criterion:forward(output, cur_ans)

			if j % opt.print_every == 0 then
			   print(string.format("Iteration %d, loss %1.6f", j, loss))
			end
			
			local grad = criterion:backward(output, cur_ans)
			for i = opt.k, k, -1 do
				grad = model[i]:backward(outputs[i+1], grad) 
			end

			if grad:norm(2) < 1e-7 then
			   break
			end
			
			model[k].gradInput = grad/torch.sqrt(grad:norm(2))
			model:updateParameters(opt.learningRate)
			train_losses[#train_losses + 1] = loss -- append the new loss
		end
	end
	return model, torch.DoubleTensor(train_losses)
end

function alternateMin(opt, model, criterion, trainDs, testDs)
	local train_losses = {}
	for i = 1, opt.k do
		table.insert(train_losses, {})
	end
	local test_losses = {}
	local iter = 1
	while true do --Figure out a stopping condition
		for i = 1, opt.k do
			local loss = {}
			model, loss = trainOneLayer(opt, trainDs, trainDs:clone(), model, criterion, i)
			train_losses[i][#train_losses[i] + 1] = loss		
			--Simple stopping criterion. Loss smaller than some small number. Can be more sophisticated!
            local tmp = train_losses[i][#train_losses[i]]
            if tmp[#tmp] < 0.01 then
                    break
            end
        end
		--Test
		local t, test_loss = test(testDs, encoder, decoder, criterion, iter)
		test_losses[#test_losses + 1] = test_loss:sum()

		-- print(string.format("Epoch %4d, test loss = %1.6f", iter, torch.mean(test_loss)))
		print(string.format("%d, %1.6f", iter, torch.mean(test_loss)))
		iter = iter + 1		
	end
	return model, train_losses, test_losses
end

function saveAll()
	print('Saving everything...')
	for i = 1, opt.k do
		torch.save(dir_name..'/weights_'..i..'.dat', m.modules[i].weight)
	end
	torch.save(dir_name..'/test_loss.dat', test_loss)

	for i = 1, opt.k do
		mean_loss = {'Mean Epochal Loss, Layer '..i,
			torch.range(1, #train_losses), 
			torch.Tensor(map(torch.mean, train_loss[i])),
			'-'}
		min_loss = {'Minimum Epochal Loss, Layer '..i,
			torch.range(1, #train_losses), 
			torch.Tensor(map(torch.min, train_loss[i])),
			'-'}
		max_loss = {'Maximum Epochal Loss, Layer '..i,
			torch.range(1, #train_losses), 
			torch.Tensor(map(torch.max, train_loss[i])),
			'-'}
		plot(mean_loss, dir_name..'/Mean_loss_layer_'..i..'.png', 'Epochs', 'Mean Loss', 'Layer '..i..'Mean Loss Plot')
		plot(min_loss, dir_name..'/Minimum_loss_layer_'..i..'.png', 'Epochs', 'Minimum Loss', 'Layer '..i..'Minimum Loss Plot')
		plot(max_loss, dir_name..'/Maximum_loss_layer_'..i..'.png', 'Epochs', 'Maximum Loss', 'Layer '..i..'Maximum Loss Plot')
	end

	tst_loss = {'Test Loss',
		torch.range(1, #test_loss),
		torch.Tensor(test_loss),
		'-'}

	plot(tst_loss, dir_name..'/Test_Loss.png', 'Epochs', 'Loss', 'Test Loss Plot')
end

model = {}
-- encoder
for i = 1, opt.k + 1 do
	layer = nn.Sequential()
	layer:add(nn.Linear(sizes[i], sizes[i+1]))
	table.insert(model, layer)
	print(layer)
end

crit = nn.MSECriterion()

trainData = load_dataset('train', opt.train_size)
testData, testLabels = load_dataset('test', opt.test_size)
m = nil
tr_loss = nil
tst_loss = nil
m, tr_loss, tst_loss= alternateMin(opt, model, crit, trainData, testData)
saveAll()


