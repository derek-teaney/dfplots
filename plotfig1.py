import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import json
import h5py as h5
import scienceplots

plt.style.use(['science', 'nature'])
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', ['#0085CA',  '#008F00', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e'])

plt.rcParams['font.size'] = 9
plt.rcParams['xtick.minor.visible'] = False
plt.rcParams['ytick.minor.visible'] = False
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['axes.labelsize'] =  9
plt.rcParams['axes.labelsize'] =  9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
# axes.labelsize: 7
# xtick.labelsize: 7
# ytick.labelsize: 7
# legend.fontsize: 7
# font.size: 7


########################################################################
fourpietabys = 12.0
gaussian_const = 0.05
gaussian_amplitude = 8.0
gaussian_width = 25.0

def getnames(fourpietabys, gaussian_const, gaussian_amplitude, gaussian_width):
    """ Helper function for getdata """
    s='figure12_data/nbys_{}_d_{}_A_{}_w_{}'.format(fourpietabys, gaussian_const, gaussian_amplitude, gaussian_width)
    return s.replace('.', 'd')

def getdata(fourpietabys,gaussian_const, gaussian_amplitude, gaussian_width, tag='Ttt'):

    variables = {'Ttt':0, 'Ttx':1, 'eps':2, 'ux':3 } 
    name = getnames(fourpietabys, gaussian_const, gaussian_amplitude, gaussian_width)
    name0 = getnames(0.0, gaussian_const, gaussian_amplitude, gaussian_width)

    data = json.load(open(name + '.json'))

    with h5.File(name + '.h5', 'r') as file:
        finaldata = file['finaldata'][:,variables[tag]]

    try :
        with h5.File(name0 + '.h5', 'r') as file:
            finaldata_ideal = file['finaldata'][:,variables[tag]]
 
    except:
        name0 = getnames(0, gaussian_const, gaussian_amplitude, gaussian_width)
        with h5.File(name0 + '.h5', 'r') as file:
            finaldata_ideal = file['finaldata'][:,variables[tag]]   
            
    # with h5.File(name0 + '.h5', 'r') as file:
    #     finaldata_ideal = file['finaldata'][:,variables[tag]]

    print(name + '.h5', '\n', name + '.json', '\n' , name0 + '.h5',  '\n', name + '_out')
    solverdata = np.loadtxt(name + '_out/{}.txt'.format(tag)) 
    x = np.loadtxt(name +'_out/x.txt') 

    return x, finaldata, solverdata[-1,:], finaldata_ideal, data

########################################################################
def plot(gaussian_const, gaussian_amplitude, gaussian_width):
    """ Basic plot not used in paper """ 
    x, df, bdnk, ideal, data = getdata(4.0, gaussian_const, gaussian_amplitude, gaussian_width)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs.flatten()
    plt.sca(axs[0])
    plt.gca().set_ylim(0.00, 1.0)
    plt.plot(x, df, label='density frame ')
    plt.plot(x, ideal, 'k', linewidth=0.5, label='Ideal Hydro')
    plt.gca().set_xlabel('$x$')
    plt.gca().set_ylabel('$T^{tt}$')

########################################################################
def plot1Presentation():
    """ Presentation version of figure 1 """
    gaussian_const = 0.12
    gaussian_amplitude = 0.48
    gaussian_width = 25.0

    x, df, bdnk, ideal, data = getdata(1.0, gaussian_const, gaussian_amplitude, gaussian_width)
    fig, axs = plt.subplots(1,1, figure=(10,10))
    #fig.suptitle('density frame vs BDNK $\eta/s = {}/4\pi$'.format(data['eta_over_s']*4.0*np.pi))
    plt.sca(axs)
    plt.gca().set_ylim(0.00, 0.325)
    plt.gca().set_xlim(-75, 75)
    plt.plot(x, df, 'C0', linewidth=1.4, label='density frame ')
    plt.plot(x, bdnk, 'C1', linewidth=1.4, linestyle="--", label='BDNK')
    # plt.plot(x, ideal, 'k:', linewidth=0.5, label='Ideal Hydro')
    plt.gca().set_xlabel('x')
    plt.gca().set_ylabel('$T^{tt}$')
    plt.legend(loc='lower left')

    x, df, bdnk, ideal, data = getdata(3.0, gaussian_const, gaussian_amplitude, gaussian_width)
    plt.plot(x, df, 'C0', linewidth=1.4, label='density frame ')
    plt.plot(x, bdnk, 'C1', linewidth=1.4, linestyle="--", label='BDNK')

    x, df, bdnk, ideal, data = getdata(6.0, gaussian_const, gaussian_amplitude, gaussian_width)
    plt.plot(x, df, 'C0', linewidth=1.4, label='density frame ')
    plt.plot(x, bdnk, 'C1', linewidth=1.4, linestyle="--", label='BDNK')

    # Set a text label with the value of eta/s
    plt.annotate(r'$4\pi\eta/s = 1, 3, 6$', (0.9,0.25), xycoords='figure fraction', ha='right')

    plt.savefig('smoothtest1_screen.pdf')

########################################################################
def plot1():
    """ Figure 1 in paper """ 
    gaussian_const = 0.12
    gaussian_amplitude = 0.48
    gaussian_width = 25.0

    x, df, bdnk, ideal, data = getdata(1.0, gaussian_const, gaussian_amplitude, gaussian_width)
    fig, axs = plt.subplots(1,1, figure=(10,10))
    #fig.suptitle('density frame vs BDNK $\eta/s = {}/4\pi$'.format(data['eta_over_s']*4.0*np.pi))
    plt.sca(axs)
    plt.gca().set_ylim(0.00, 0.325)
    plt.gca().set_xlim(-75, 75)
    plt.plot(x, df, 'C0', linewidth=1.2, label='density drame ')
    plt.plot(x, bdnk, 'C3', linewidth=1.2, linestyle="--", label='BDNK')
    plt.plot(x, ideal, 'k--', linewidth=0.5, label='ideal hydro')
    plt.gca().set_xlabel('x')
    plt.gca().set_ylabel('$T^{tt}$')
    plt.legend(loc='lower left')

    x, df, bdnk, ideal, data = getdata(3.0, gaussian_const, gaussian_amplitude, gaussian_width)
    plt.plot(x, df, 'C0', linewidth=1.0, label='density frame ')
    plt.plot(x, bdnk, 'C3', linewidth=1.0, linestyle="--", label='BDNK')

    x, df, bdnk, ideal, data = getdata(6.0, gaussian_const, gaussian_amplitude, gaussian_width)
    plt.plot(x, df, 'C0', linewidth=1.0, label='density frame ')
    plt.plot(x, bdnk, 'C3', linewidth=1.0, linestyle="--",  label='BDNK')

    # Set a text label with the value of eta/s
    plt.annotate(r'$4\pi\eta/s = 0, 1, 3, 6$', (0.9,0.25), xycoords='figure fraction', ha='right')

    plt.savefig('smoothtest1.pdf')

########################################################################
def plot1b():
    """ Figure 1b in paper """ 
    gaussian_const = 0.12
    gaussian_amplitude = 0.48
    gaussian_width = 25.0

    x, df, bdnk, ideal, data = getdata(20.0, gaussian_const, gaussian_amplitude, gaussian_width)
    xfree, Tttfree, Ttxfree = freefunction(47.5, gaussian_const, gaussian_amplitude, gaussian_width)

    fig, axs = plt.subplots(1,1, figure=(10,10))
    #fig.suptitle('density frame vs BDNK $\eta/s = {}/4\pi$'.format(data['eta_over_s']*4.0*np.pi))
    plt.sca(axs)
    plt.vlines(x=47.5+2.5*5., ymin=0.12-0.025, ymax=0.12 + 0.025, color='0.8', linewidth=4.0)
    plt.vlines(x=-47.5-2.5*5., ymin=0.12-0.025, ymax=0.12 + 0.025,  color='0.8', linewidth=4.0)

    plt.gca().set_ylim(0.00, 0.325)
    plt.gca().set_xlim(-75, 75)
    plt.plot(x, df, 'C0', linewidth=1.2)
    plt.plot(x, bdnk, 'C3', linewidth=1.2, linestyle="--")
    plt.plot(xfree, Tttfree, 'k--', linewidth=0.5, label='free streaming')
    plt.gca().set_xlabel('x')
    plt.gca().set_ylabel('$T^{tt}$')
    plt.legend(loc="upper left")
    # plt.legend(frameon=True, loc="upper left",fancybox=False, framealpha=0.8, edgecolor='white')

    plt.annotate(r'$4\pi\eta/s = 20$', (0.9, 0.25), xycoords='figure fraction',  ha='right')
    plt.savefig('smoothtest1b.pdf')


def plot1bPresentation():
    """ This is the presentation version """ 
    gaussian_const = 0.12
    gaussian_amplitude = 0.48
    gaussian_width = 25.0

    x, df, bdnk, ideal, data = getdata(20.0, gaussian_const, gaussian_amplitude, gaussian_width)
    fig, axs = plt.subplots(1,1, figure=(10,10))
    #fig.suptitle('density frame vs BDNK $\eta/s = {}/4\pi$'.format(data['eta_over_s']*4.0*np.pi))
    plt.sca(axs)
    plt.vlines(x=60, ymin=0.12-0.025, ymax=0.12 + 0.025, color='0.8', linewidth=4.0)
    plt.vlines(x=-60, ymin=0.12-0.025, ymax=0.12 + 0.025,  color='0.8', linewidth=4.0)

    plt.gca().set_ylim(0.00, 0.325)
    plt.gca().set_xlim(-75, 75)
    plt.plot(x, df, 'C0', linewidth=1.4, label='density frame ')
    plt.plot(x, bdnk, 'C1', linewidth=1.4, linestyle="--", label='BDNK')
    # plt.plot(x, ideal, 'k', linewidth=0.5, label='Ideal Hydro')
    plt.gca().set_xlabel('x')
    plt.gca().set_ylabel('$T^{tt}$')
    #plt.legend()

    plt.annotate(r'$4\pi\eta/s = 20$', (0.9, 0.25), xycoords='figure fraction',  ha='right')
    plt.savefig('smoothtest1b_screen.pdf')


########################################################################
def freefunction(finaltime, gaussian_const = 0.12, gaussian_amplitude = 0.48, gaussian_width = 25.0):
    """ Returns for the free theory

        x[:], Ttt[:], Ttx[:]

    """
    x=np.linspace(-75.1,75.1,500)
    delta=gaussian_const
    A=gaussian_amplitude
    w=np.sqrt(gaussian_width)

    tc=finaltime
    analytic=delta+np.sqrt(np.pi)/2*A*w/(2*tc)*(sp.erf((x+tc)/w)-sp.erf((x-tc)/w))
    analyticflux= A*w/(4*tc**2)*( w*( np.exp(-(tc+x)**2/w**2  ) - np.exp(-(tc-x)**2/w**2) ) +   np.sqrt(np.pi)*x*(sp.erf((x+tc)/w)-sp.erf((x-tc)/w))  )

    return  x, analytic, analyticflux

########################################################################
def plot2():
    """ This is the same as figure one but with different parameters""" 
    gaussian_const = 0.06
    gaussian_amplitude = 9.6
    gaussian_width = 25.0

    # x, df, bdnk, ideal, data = getdata(3.0, gaussian_const, gaussian_amplitude, gaussian_width)
    fig, axs = plt.subplots(1,1, figure=(10,10))
    #fig.suptitle('density frame vs BDNK $\eta/s = {}/4\pi$'.format(data['eta_over_s']*4.0*np.pi))
    plt.sca(axs)
    #plt.gca().set_ylim(0.00, 0.28)
    plt.gca().set_xlim(-75, 75)

    x, df, bdnk, ideal, data = getdata(1.0, gaussian_const, gaussian_amplitude, gaussian_width)
    #plt.plot(x, df, 'C0--', label='density frame ')
    plt.plot(x, bdnk, 'C3',linestyle=(0,(2,1)), label='BDNK')
    #plt.plot(x, ideal, 'k', linewidth=0.5, label='Ideal hydro')

    x, df, bdnk, ideal, data = getdata(3.0, gaussian_const, gaussian_amplitude, gaussian_width)
    plt.plot(x, df, 'C0')
    plt.plot(x, bdnk, 'C3', linestyle=(0,(2,1)))

    x, df, bdnk, ideal, data = getdata(6.0, gaussian_const, gaussian_amplitude, gaussian_width)
    plt.plot(x, df, 'C0')
    plt.plot(x, bdnk, 'C3', linestyle=(0,(2,1)))
    # plt.axvline(x=61,  ymax=0.2, color='0.7', linewidth=1.0)
    # plt.axvline(x=-61, ymax=0.2,  color='0.7', linewidth=1.0)

    # x, df, bdnk, ideal, data = getdata(10.0, gaussian_const, gaussian_amplitude, gaussian_width)
    # plt.plot(x, df, 'C0')
    # plt.plot(x, bdnk, 'C1', linestyle=(0,(2,2)))


    # x, df, bdnk, ideal, data = getdata(10.0, gaussian_const, gaussian_amplitude, gaussian_width)
    # plt.plot(x, df, 'C0')
    # plt.plot(x, bdnk, 'C1')
    plt.gca().set_xlabel('x')
    plt.gca().set_ylabel('$T^{tt}$')

    # # # Set a text label with the value of eta/s
    # plt.text(0.05, 0.05, r'$4\pi\eta/s = 3, 10$', fontsize=7)

    plt.legend(loc='upper center', title=r'$4\pi\eta/s=0,1,3,6$', handlelength=0.75)

    plt.savefig('smoothtest2.pdf')

########################################################################
def plot2b():
    """  Plots the velocity vs position for test2 """
    gaussian_const = 0.06
    gaussian_amplitude = 9.6
    gaussian_width = 25.0

    # x, df, bdnk, ideal, data = getdata(3.0, gaussian_const, gaussian_amplitude, gaussian_width)
    fig, axs = plt.subplots(1,1, figure=(10,10))
    #fig.suptitle('density frame vs BDNK $\eta/s = {}/4\pi$'.format(data['eta_over_s']*4.0*np.pi))
    plt.sca(axs)
    #plt.gca().set_ylim(0.00, 0.28)
    plt.gca().set_ylim(-1.4, 1.4)
    plt.gca().set_xlim(-75, 75)

    plt.axvline(x=60, ymin=0.43, ymax=0.57, color='0.7', linewidth=4.0)
    plt.axvline(x=-60, ymin=0.43, ymax=0.57,  color='0.7', linewidth=4.0)

    x, df, bdnk, ideal, data = getdata(10.0, gaussian_const, gaussian_amplitude, gaussian_width, tag='ux')
    plt.plot(x, df, 'C0', linewidth=1.2,  label='density frame ')
    plt.plot(x, bdnk, 'C3',linewidth=1.2, linestyle="--", label='BDNK')


    plt.gca().set_xlabel('x')
    plt.gca().set_ylabel('$u^x$')


    # # # Set a text label with the value of eta/s
    # plt.text(0.05, 0.05, r'$4\pi\eta/s = 3, 10$', fontsize=7)
    #plt.legend(title=r'$4\pi\eta/s=10$')
    plt.legend(loc='upper left', title=r'$4\pi\eta/s=10$')

    plt.savefig('smoothtest2b.pdf')

########################################################################
def plot2c():
    """  Plots the Ttx vs position for test2 """
    gaussian_const = 0.06
    gaussian_amplitude = 9.6
    gaussian_width = 25.0

    # x, df, bdnk, ideal, data = getdata(3.0, gaussian_const, gaussian_amplitude, gaussian_width)
    fig, axs = plt.subplots(1,1, figure=(10,10))
    #fig.suptitle('density frame vs BDNK $\eta/s = {}/4\pi$'.format(data['eta_over_s']*4.0*np.pi))
    plt.sca(axs)
    #plt.gca().set_ylim(0.00, 0.28)
    plt.gca().set_xlim(-75, 75)
    plt.axvline(x=60, ymin=0.43, ymax=0.57, color='0.7', linewidth=4.0)
    plt.axvline(x=-60, ymin=0.43, ymax=0.57,  color='0.7', linewidth=4.0)


    x, df, bdnk, ideal, data = getdata(10.0, gaussian_const, gaussian_amplitude, gaussian_width, tag='Ttx')
    plt.plot(x, df, 'C0',  linewidth=1.2, label='density frame')
    plt.plot(x, bdnk, 'C3',linewidth=1.2, linestyle="--",label='BDNK')
    plt.savefig('smoothtest2c.pdf')
    plt.gca().set_xlabel('x')
    plt.gca().set_ylabel('$T^{tx}$')

    # # # Set a text label with the value of eta/s
    # plt.text(0.05, 0.05, r'$4\pi\eta/s = 3, 10$', fontsize=7)
    # plt.legend(title=r'$4\pi\eta/s=10$')
    plt.gca().set_xlabel('x')
    plt.gca().set_ylabel('$T^{tx}$')
    plt.legend(title=r'$4\pi\eta/s=10$', loc='upper left')
    plt.savefig('smoothtest2c.pdf')

def plot2d():
    gaussian_const = 0.06
    gaussian_amplitude = 9.6
    gaussian_width = 25.0

    # x, df, bdnk, ideal, data = getdata(3.0, gaussian_const, gaussian_amplitude, gaussian_width)
    fig, axs = plt.subplots(1,1, figure=(10,10))
    #fig.suptitle('density frame vs BDNK $\eta/s = {}/4\pi$'.format(data['eta_over_s']*4.0*np.pi))
    #plt.sca(axs)
    ##plt.gca().set_ylim(0.00, 0.28)
    #plt.gca().set_xlim(-75, 75)
    #plt.axvline(x=60, ymin=0.43, ymax=0.57, color='0.7', linewidth=4.0)
    #plt.axvline(x=-60, ymin=0.43, ymax=0.57,  color='0.7', linewidth=4.0)


    x, df, bdnk, ideal, data = getdata(10.0, gaussian_const, gaussian_amplitude, gaussian_width, tag='Ttt')
    plt.plot(x, df, 'C0',  label='density frame ')
    plt.plot(x, bdnk, 'C3',linestyle=(0,(2,1)), linewidth=1.0, label='BDNK')
    #plt.savefig('smoothtest2c.pdf')
    plt.gca().set_xlabel('x')
    plt.gca().set_ylabel('$T^{tt}$')
    plt.show()

    # # # Set a text label with the value of eta/s
    # plt.text(0.05, 0.05, r'$4\pi\eta/s = 3, 10$', fontsize=7)
    # plt.legend(title=r'$4\pi\eta/s=10$')
    #plt.gca().set_xlabel('x')
    #plt.gca().set_ylabel('$T^{tx}$')
    #plt.legend(title=r'$4\pi\eta/s=10$', loc='upper left')
    #plt.savefig('smoothtest2d.pdf')


#plot1Presentation()
#plot1bPresentation()
plot1()
plot1b()
plot2()
plot2b()
plot2c()
#plot2d()

#plot2(gaussian_const, gaussian_amplitude, gaussian_width)
