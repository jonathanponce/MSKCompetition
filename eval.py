import mxnet as mx
import data_helpers
import numpy as np
import re

symbol, arg_params, aux_params = mx.model.load_checkpoint("checkpoint/checkpoint", 20)
batch_size=50
model = mx.model.FeedForward(symbol=symbol, ctx=mx.cpu(), arg_params=arg_params, aux_params=aux_params, numpy_batch_size=batch_size)
x, y, vocab, vocab_inv  = data_helpers.load_data_eval()
eval = mx.io.NDArrayIter(data=x,batch_size=batch_size, shuffle=False)
result = model.predict(eval)
labels = open("test_variants/test_variants").read().splitlines()
labels.pop(0)
with open("submission", "a") as myfile:
    myfile.write("ID,class1,class2,class3,class4,class5,class6,class7,class8,class9\n")
    for x in range(len(result)):
       myfile.write(str(x)+","+('%.2f' % result[x][0])+","+('%.2f' % result[x][1])+","+('%.2f' % result[x][2])+","+('%.2f' % result[x][3])+","+('%.2f' % result[x][4])+","+('%.2f' % result[x][5])+","+('%.2f' % result[x][6])+","+('%.2f' % result[x][7])+","+('%.2f' % result[x][8])+"\n")