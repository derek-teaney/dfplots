import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import h5py as h5
import json
import scienceplots
import matplotlib.colors as mcolors


# Set parameters to make it look like gnuplot
plt.style.use(['science', 'nature'])

plt.rcParams['axes.prop_cycle'] = plt.cycler('color', ['#0085CA', '#008F00', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e'])

plt.rcParams['font.size'] = 8
plt.rcParams['xtick.minor.visible'] = False
plt.rcParams['ytick.minor.visible'] = False
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['axes.labelsize'] =  8
plt.rcParams['axes.labelsize'] =  8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8



################################################################################
def getEKTdata(lambdaekt,fourpietabys, tag='Ttt', testcase='test1'):
    """ Returns the EKT data. The output is 

    x[:], TttEKT[:], TttDF[:], finaltime 

    If tag='Ttx' this function returns  Ttx instead of Ttt
    """

    Nz=150
    nug=16

    variables = {'Ttt':8, 'Ttx':11}

    EKTdata=np.loadtxt("./new_EKT/%s_L%d_gluon_Tmunu_vs_time.out" % (testcase, lambdaekt))
    nx,ny = EKTdata.shape
    EKTdata.shape=(nx//Nz,Nz,ny)
    sl=-6
    xekt = EKTdata[0,:,18] 


    # Get Either the  Ttt component or Ttx component
    Tttekt = nug*EKTdata[sl,:,variables[tag]]
    return xekt, Tttekt

def getEKTdata(lambdaekt,fourpietabys, tag='Ttt'):
    """ Returns the EKT data. The output is 

    x[:], TttEKT[:]
    """

    Nz=150
    nug=16

    EKTdata=np.loadtxt("./new_EKT/test1_L%d_gluon_Tmunu_vs_time.out" % lambdaekt)
    nx,ny = EKTdata.shape
    EKTdata.shape=(nx//Nz,Nz,ny)
    sl=-1
    xekt = EKTdata[0,:,18] 
    Tttekt = nug*EKTdata[sl,:,8]

    return xekt, Tttekt


################################################################################
def getnames(fourpietabys, gaussian_const, gaussian_amplitude, gaussian_width):
    """ Return the filename for a gvien value of the paramaeters. This a helper function used by getDFdata """
    s='DFAndBDNK/nbys_{}_d_{}_A_{}_w_{}'.format(fourpietabys, gaussian_const, gaussian_amplitude, gaussian_width)
    return './' + s.replace('.', 'd')

def getDFdata(fourpietabys,gaussian_const, gaussian_amplitude, gaussian_width, tag='Ttt'):

    """  Returns the density frame and BDNK data for a given run 


    The output is x[:], TttDF[:], TttBDNK[:], TttIdeal[:], info


    The info is an associative array containing the parameters of the run.
    """

    variables = {'Ttt':0, 'Ttx':1, 'eps':2, 'ux':3 } 
    name = getnames(fourpietabys, gaussian_const, gaussian_amplitude, gaussian_width)
    name0 = getnames(0.0, gaussian_const, gaussian_amplitude, gaussian_width)

    print(name, name0)
    data = json.load(open(name + '.json'))

    with h5.File(name + '.h5', 'r') as file:
        finaldata = file['finaldata'][:,variables[tag]]

    try :
        with h5.File(name0 + '.h5', 'r') as file:
            finaldata_ideal = file['solution'][-1,:,variables[tag]]
    except:
        name0 = getnames(0, gaussian_const, gaussian_amplitude, gaussian_width)
        with h5.File(name0 + '.h5', 'r') as file:
            finaldata_ideal = file['finaldata'][:,variables[tag]]   
            
    with h5.File(name0 + '.h5', 'r') as file:
        finaldata_ideal = file['finaldata'][:,variables[tag]]

    solverdata = np.loadtxt(name + '_out/{}.txt'.format(tag)) 
    x = np.loadtxt(name +'_out/x.txt') 

    return x, finaldata, solverdata[-1,:], finaldata_ideal, data


################################################################################
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
def plotIC(case='DF', lambda_case=0):
    """Plots the initial conditions """ 

    listetas=[0.180, 0.513, 1.48]
    listlambda=[20, 10, 5]

    etas4pi=4*np.pi*np.array(listetas)

    gaussian_const = 0.12
    gaussian_amplitude = 0.48
    gaussian_width = 25.0

    fig1, ax1 = plt.subplots()
    ax1.set_xlim(-15, 15)
    #ax1.set_xticks([-75,-50,-25,0,25,50,75])
    ax1.set_ylim(-0.1, 0.65)

    i = lambda_case

    xfree0, freeTtt0, freeTtx0 = freefunction(0.000001)


    if case == 'DF':
        ax1.plot(xfree0, freeTtt0, "C0", linewidth=1.2, label='Density Frame') 
    else:
        ax1.plot(xfree0, freeTtt0,  "C0", linewidth=1.2, label='BDNK') 

    ax1.plot(xfree0, freeTtt0, "C3", linestyle=(0,(3,2)), linewidth=1.2, label='QCD kinetics')

    # ax1.plot(x, ideal, "k--", linewidth=0.5, label=r"$\eta/s=0$ and $\infty$") 
    # ax1.plot(xfree, freeTtt, "k--", linewidth=0.5)

    ax1.legend(frameon=True,loc="lower left", fancybox=False, framealpha=0.8, edgecolor='white')
    # # Set a text label with the value of eta/s
    ax1.annotate('initial conditions', (0.85,0.25), xycoords='figure fraction', ha='right',bbox=dict(alpha=0.8,facecolor='white',edgecolor='white'))

    ax1.annotate(r'$A=0.48\;{\rm GeV}$'+"\n" + r"$\delta=0.12\; {\rm GeV}$" + "\n" + r"$L=5 \;{\rm GeV}^{-1}$", xy=(0.7,0.7), xycoords='figure fraction', linespacing=1.5)

    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$T^{tt}$')
    fig1.tight_layout() 

    #name = "{}_{}.pdf".format(case,listlambda[lambda_case])
    fig1.savefig("test1_ic.pdf")
    #fig1.savefig(name)


########################################################################
def plotStress(case='DF', lambda_case=0):
    """Makes a nice comparison between the density frame and kintec theory or BDNK and kinetic theory """

    finaltime=47.5
    gaussian_const = 0.12
    gaussian_amplitude = 0.48
    gaussian_width = 25.0

    listetas=[0.180, 0.513, 1.48]
    etas4pi=4*np.pi*np.array(listetas)
    listlambda=[20, 10, 5]

    fig1, ax1 = plt.subplots()
    ax1.set_xlim(-75, 75)
    ax1.set_xticks([-75,-50,-25,0,25,50,75])
    ax1.set_ylim(0.0, 0.50)

    i = lambda_case

    # Get the kinetic theory data
    et = listetas[i] 
    xekt, Tttekt = getEKTdata(listlambda[i], et)

    # Get density frame, bdnk, and idealhydro
    x, df, bdnk, ideal, data = getDFdata(etas4pi[i], gaussian_const, gaussian_amplitude, gaussian_width)

    # Get Free streaming data
    xfree, freeTtt, freeTtx = freefunction(finaltime)


    if case == 'DF':
        ax1.plot(x, df, "C0", linewidth=1.2, label='Density Frame') 
    else:
        ax1.plot(x, bdnk, "C0", linewidth=1.2, label='BDNK') 

    ax1.plot(xekt, Tttekt, "C3", linestyle=(0,(3,2)), linewidth=1.2, label='QCD kinetics')

    ax1.plot(x, ideal, "k--", linewidth=0.5, label=r"$\eta/s=0$ and $\infty$") 
    ax1.plot(xfree, freeTtt, "k--", linewidth=0.5)

    # ax1.legend(frameon=True,loc="lower left", fancybox=False, framealpha=0.8, edgecolor='white')
    # Set a text label with the value of eta/s
    ax1.annotate(r'$4\pi\eta/s={:.1f}$'.format(4.0*np.pi*et), (0.85,0.25), xycoords='figure fraction', ha='right',bbox=dict(alpha=0.8,facecolor='white',edgecolor='white'))


    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$T^{tt}$')
    fig1.tight_layout() 

    name = "test1_{}_{}.pdf".format(case,listlambda[lambda_case])
    fig1.savefig(name)
        
################################################################################

def plotKTPlot1(case='DF'):
    """Makes a nice comparison between the density frame and kintec theory for moderate coupling """

    finaltime=47.5
    gaussian_const = 0.12
    gaussian_amplitude = 0.48
    gaussian_width = 25.0

    listetas=[0.180, 0.513, 1.48]
    etas4pi=4*np.pi*np.array(listetas)
    listlambda=[20, 10, 5]

    fig1, ax1 = plt.subplots()
    ax1.set_xlim(-75, 75)
    ax1.set_xticks([-75,-50,-25,0,25,50,75])
    ax1.set_ylim(0.0, 0.32)

    for i in range(0,2):

        # Get the kinetic theory data
        et = listetas[i] 
        xekt, Tttekt = getEKTdata(listlambda[i], et)

        # Get density frame, bdnk, and idealhydro
        x, df, bdnk, ideal, data = getDFdata(etas4pi[i], gaussian_const, gaussian_amplitude, gaussian_width)

        # Get Free streaming data
        xfree, freeTtt, freeTtx = freefunction(finaltime)


        # For the first plot add the label
        if i == 0: 
            if case == 'DF':
                ax1.plot(x, df, "C0", linewidth=1.2, label='density frame') 
            else:
                ax1.plot(x, bdnk, "C0", linewidth=1.2, label='BDNK') 

            ax1.plot(xekt, Tttekt, "C3", linewidth=1.2, linestyle="--", label='QCD kinetics')
        else:
            if case == 'DF':
                ax1.plot(x, df, "C0", linewidth=1.2) 
            else:
                ax1.plot(x, bdnk, "C0", linewidth=1.2) 

            ax1.plot(xekt, Tttekt, "C3", linewidth=1.2, linestyle="--")

    ax1.plot(x, ideal, "k:", linewidth=1.0, label="ideal hydro") 
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$T^{tt}$')

    ax1.annotate(r'$4\pi\eta/s=0, {:.1f}, {:.1f}$'.format(4.0*np.pi*listetas[0], 4.0*np.pi*listetas[1]), (0.85,0.25), xycoords='figure fraction', ha='right',bbox=dict(alpha=0.8,facecolor='white',edgecolor='white'))

    ax1.legend(loc="lower left")
    fig1.tight_layout() 

    fig1.savefig('KTPlot1.pdf')


def plotKTPlot1b(case='DF'):
    """Makes a nice comparison between the density frame and kintec theory for moderate coupling """

    finaltime = 47.5
    gaussian_const = 0.12
    gaussian_amplitude = 0.48
    gaussian_width = 25.0

    listetas=[0.180, 0.513, 1.48]
    etas4pi=4*np.pi*np.array(listetas)
    listlambda=[20, 10, 5]

    fig1, ax1 = plt.subplots()
    ax1.set_xlim(-75, 75)
    ax1.set_xticks([-75,-50,-25,0,25,50,75])
    ax1.set_ylim(0.0, 0.32)


    i= 2
    # Get the kinetic theory data
    et = listetas[i] 
    xekt, Tttekt = getEKTdata(listlambda[i], et)

    # Get density frame, bdnk, and idealhydro
    x, df, bdnk, ideal, data = getDFdata(etas4pi[i], gaussian_const, gaussian_amplitude, gaussian_width)

    # Get Free streaming data
    xfree, freeTtt, freeTtx = freefunction(finaltime)


    if case == 'DF':
        ax1.plot(x, df, "C0", linewidth=1.2, label='density frame') 
    else:
        ax1.plot(x, bdnk, "C0", linewidth=1.2, label='BDNK') 

    ax1.plot(xekt, Tttekt, "C3", linewidth=1.2, linestyle="--", label='QCD kinetics')

    ax1.plot(xfree, freeTtt, "k--", linewidth=0.5, label="free streaming") 

    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$T^{tt}$')

    ax1.annotate(r'$4\pi\eta/s={:.1f}$'.format(4.0*np.pi*listetas[2]), (0.85,0.25), xycoords='figure fraction', ha='right',bbox=dict(alpha=0.8,facecolor='white',edgecolor='white'))

    ax1.legend(loc="lower left")

    fig1.savefig('KTPlot1b.pdf')

# plotIC()
# plotStress(lambda_case=0)
# plotStress(lambda_case=1)
# plotStress(lambda_case=2)
# plotStress(case='BDNK', lambda_case=0)
# plotStress(case='BDNK', lambda_case=1)
# plotStress(case='BDNK', lambda_case=2)

plotKTPlot1()
plotKTPlot1b()

