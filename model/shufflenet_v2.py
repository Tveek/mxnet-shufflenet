import mxnet as mx

def shuffleUnit(residual, in_channels, out_channels, split):
    # for guideline 1
    equal_channels = out_channels / 2

    if split==True:
        DWConv_stride = 1
        # split feature map
        branch1 = mx.sym.slice_axis(residual, axis=1, begin=0, end=in_channels / 2)
        branch2 = mx.sym.slice_axis(residual, axis=1, begin=in_channels / 2, end=in_channels)
    else:
        DWConv_stride = 2
        branch1 = residual
        branch2 = residual

        branch1 = mx.sym.Convolution(data=branch1, num_filter=in_channels, kernel=(3, 3),
                                     pad=(1, 1), stride=(DWConv_stride, DWConv_stride), num_group=in_channels)
        branch1 = mx.sym.BatchNorm(data=branch1)

        branch1 = mx.sym.Convolution(data=branch1, num_filter=equal_channels,
                                     kernel=(1, 1), stride=(1, 1))
        branch1 = mx.sym.BatchNorm(data=branch1)
        branch1 = mx.sym.Activation(data=branch1, act_type='relu')



    branch2 = mx.sym.Convolution(data=branch2, num_filter=equal_channels,
    	              kernel=(1, 1), stride=(1, 1))
    branch2 = mx.sym.BatchNorm(data=branch2)
    branch2 = mx.sym.Activation(data=branch2, act_type='relu')


    branch2 = mx.sym.Convolution(data=branch2, num_filter=equal_channels, kernel=(3, 3),
    	               pad=(1, 1), stride=(DWConv_stride, DWConv_stride), num_group=equal_channels)
    branch2 = mx.sym.BatchNorm(data=branch2)

    branch2 = mx.sym.Convolution(data=branch2, num_filter=equal_channels,
    	               kernel=(1, 1), stride=(1, 1))
    branch2 = mx.sym.BatchNorm(data=branch2)
    branch2 = mx.sym.Activation(data=branch2, act_type='relu')

    data = mx.sym.concat(branch1, branch2, dim=1)
    data = mx.contrib.sym.ShuffleChannel(data=data, group=2)

    return data

def make_stage(data, stage, multiplier=1):
    stage_repeats = [3, 7, 3]

    if multiplier == 0.5:
        out_channels = [-1, 24, 48, 96, 192]
    elif multiplier == 1:
        out_channels = [-1, 24, 116, 232, 464]
    elif multiplier == 1.5:
        out_channels = [-1, 24, 176, 352, 704]
    elif multiplier == 2:
        out_channels = [-1, 24, 244, 488, 976]

    # DWConv_stride = 2
    data = shuffleUnit(data, out_channels[stage - 1], out_channels[stage],
    	               split=False)
    # DWConv_stride = 1
    for i in range(stage_repeats[stage - 2]):
        data = shuffleUnit(data, out_channels[stage], out_channels[stage],
        	               split=True)

    return data

def get_symbol(num_classes=1000, **kwargs):
    data = mx.sym.var('data')
    data = mx.sym.Convolution(data=data, num_filter=24,
        	                  kernel=(3, 3), stride=(2, 2), pad=(1, 1))
    data = mx.sym.BatchNorm(data=data)
    data = mx.sym.Activation(data=data, act_type='relu')

    data = mx.sym.Pooling(data=data, kernel=(3, 3), pool_type='max',
    	                  stride=(2, 2), pad=(1, 1))

    data = make_stage(data, 2)

    data = make_stage(data, 3)

    data = make_stage(data, 4)

    extra_conv = mx.sym.Convolution(data=data, num_filter=1024,
                                              kernel=(1, 1), stride=(1, 1))
    extra_conv = mx.sym.BatchNorm(data=extra_conv)
    data = mx.sym.Activation(data=extra_conv, act_type='relu')

    data = mx.sym.Pooling(data=data, kernel=(1, 1), global_pool=True, pool_type='avg')

    data = mx.sym.flatten(data=data)

    data = mx.sym.FullyConnected(data=data, num_hidden=num_classes)

    out = mx.sym.SoftmaxOutput(data=data, name='softmax')

    return out
