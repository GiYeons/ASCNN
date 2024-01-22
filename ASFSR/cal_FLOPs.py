
'''
=================== READ ME ==========================
used to calculate the FLOPs of each model
we need kernel size (ker), input image size (lr or hr), input channels (cin), and output channels (cout) to calculate
FLOPs of a layer = (ker * ker) * lr * cin * cout
=================== READ ME ==========================
'''

# round up
def roundn(n, p=6):
    n = int(round(n, -p))
    n = int(str(n)[:-p])
    return n


# we use the reduction provided in their paper to calculate the overall FLOPs
def tssrn(h = 720, w = 1280, flops = True, scale = 2, reduction=0):

    hr = h * w
    lr = (h / scale) * (w / scale)
    mult = sum([(5*5) * lr * 32,
               lr * 32* 16,
               (3*3) * lr * 16*16*4,
               lr* 16 * 32,
               (9*9)* hr * 32])

    result = mult * (1-reduction)
    result = roundn(result)
    print(result)

def fsrcnn(h = 720, w = 1280, flops = True, scale = 2):

    hr = h * w
    lr = (h / scale) * (w / scale)
    mult = sum([(5*5) * lr * 56,
               lr * 56* 12,
               (3*3) * lr * 12*12*4,
               lr* 12 * 56,
               (9*9)* hr * 56])

    result = mult
    result = roundn(result)
    print(result)

def fsrcnn_s(h = 720, w = 1280, flops = True, scale = 2):
    hr = h * w
    lr = (h / scale) * (w / scale)

    if flops == False:
        lr , hr = 1, 1
    mult = sum([(5*5) * lr * 32,
               lr * 32* 5,
               (3*3) * lr * 5*5*1,
               lr* 5 * 32,
               (9*9)* hr * 32])

    result = mult
    result = roundn(result)
    print(result)

'''
hpf : high parameter filters
lpf : low parameter filters

r: reduction rate
spar : sparsity (sparsity determines how much belongs to the lpf and how much belongs to the hpf)
we calculate their FLOPs on each layer
'''
def hpf(k, lr, cin, cout):
    return (k*k)*lr* cin*cout

def lpf(k, lr, cin, cout, r):
    return sum([(k*k) * lr * cin * (cout/r),
               lr * (cout/r) * cout])

#reduction before the last convolution (this doesn't have any skip)
def lpf_last_reduce(lr, cin, r):
    return lr*cin*(cin/r)

# last convolution of the lpf (this has skip)
def lpf_last_conv(k, lr, cin, cout, r):
    return (k*k) *lr*(cin/r)*cout



def asfsr(h = 720, w = 1280, flops = True, scale = 2, spar = 0.0, r = 4):
    avg = 0

    hr = h * w
    lr = (h / scale) * (w / scale)

    if flops == False:
        lr , hr = 1, 1

    high = sum([hpf(5, lr, 1, 32),
               hpf(1, lr, 32, 16),
               hpf(3, lr, 16, 16)*4, # 4 layers
               hpf(1, lr, 16, 32),
               hpf(9, hr, 32, 1)]) * (1-spar)

    low = sum([lpf(5, lr, 1, 32, r),
               lpf(1, lr, 32, 16, r),
               lpf(3, lr, 16, 16, r)*4,
               lpf(1, lr, 16, 32, r),
               lpf_last_conv(9, hr, 32, 1, r)]) * (spar)

    # this part has no skipping
    comn=0
    if spar !=0:
        comn = sum([lpf_last_reduce(lr, 32, r), # before transposed convolution
                    lr]) # mask generation : FLOPs due to averaging (division)

    result = comn + high + low

    result =roundn(result)
    print(result)

def espcn(h = 720, w = 1280, r=4, flops = True, scale = 2, spar = 0.0):
    avg = 0

    hr = h * w
    lr = (h / scale) * (w / scale)

    if flops == False:
        lr, hr = 1, 1

    high = sum([hpf(5, lr, 1, 64),
                hpf(3, lr, 64, 32),
               hpf(3, lr, 32, (scale**2))])* (1 - spar)

    low = sum([lpf(5, lr, 1, 64, r),
                lpf(3, lr, 64, 32, r),
               lpf_last_conv(3, lr, 32, (scale**2), r)])* (spar)

    comn=0
    if spar !=0:
        comn = sum([lpf_last_reduce(lr, 32, r), # before transposed convolution
                    lr]) #mask generation

    result = comn + high + low

    # result = roundn(result)
    print(result)

def srcnn(h = 720, w = 1280, r=4, flops = True, scale = 2, spar = 0.0):
    avg = 0

    hr = h * w
    lr = (h / scale) * (w / scale)

    if flops == False:
        lr, hr = 1, 1

    high = sum([hpf(9, hr, 1, 64),
                hpf(5, hr, 64, 32),
               hpf(5, hr, 32, 1)])* (1 - spar)

    low = sum([lpf(9, hr, 1, 64, r),
                lpf(5, hr, 64, 32, r),
               lpf_last_conv(5, hr, 32, 1, r)])* (spar)

    comn=0
    if spar !=0:
        comn = sum([lpf_last_reduce(hr, 32, r), # before transposed convolution
                    lr]) #mask generation

    mult = comn + high + low

    result = roundn(mult)
    print(result)




asfsr(scale=2, spar=0.2)


# exit()

# spars = [0, 0.2, 0.54, 0.73, 0.84]
# spars = [0, 0.17, 0.48, 0.68, 0.81]
# spars = [0, 0.14, 0.43, 0.64, 0.79]
# spars = [0, 0.45]
spars = [0.2, 0.54, 0.73]
for spar in spars:
    asfsr(spar=spar, scale = 2, r=4)
#
# asfsr(spar=0)
# asfsr() = [0.45, 0.61, 0.73, 0.81]
# for spar in spars:
#     espcn(spar=spar, r=16)
# #
#
# srcnn(spar=0)
# spars = [0.45, 0.61, 0.73, 0.81]
# for spar in spars:
#     srcnn(spar=spar, r=16)