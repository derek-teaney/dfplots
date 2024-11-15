import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import h5py as h5
import json
import scienceplots
import matplotlib.colors as mcolors

# Set parameters to make it look like gnuplot
plt.style.use(['science', 'nature'])

plt.rcParams['axes.prop_cycle'] = plt.cycler('color', ['#0085CA', '#008F00', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e', mcolors.CSS4_COLORS["salmon"]])

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
def getEKTdata(lambdaekt, testcase='test1', tag='Ttt'):
    """ Returns the EKT data. The output is 

    x[:], TttEKT[:], TttDF[:], finaltime 

    If tag='Ttx' this function returns  Ttx instead of Ttt

    lambdaekt should be one of 20,10,5 as is listed in list_lambdaekt
    """

    Nz=150
    nug=16

    variables = {'Ttt':8, 'Ttx':11}

    EKTdata=np.loadtxt("./new_EKT/%s_L%d_gluon_Tmunu_vs_time.out" % (testcase, lambdaekt))
    nx,ny = EKTdata.shape

    print(nx,ny)
    EKTdata.shape=(nx//Nz,Nz,ny)
    print(EKTdata.shape)
    sl=-1
    xekt = EKTdata[0,:,18] 


    # Get Either the  Ttt component or Ttx component
    Tttekt = nug*EKTdata[sl,:,variables[tag]]

    return xekt, Tttekt


################################################################################
def getMUSICdata(visMUSIC, testcase='test1', tag='Ttt', tfinal=50, shear_relax_factor='5'):
    """ Returns the MUSIC data. The output is 

    x[:], TttMUSIC[:] 

    If tag='Ttx' this function returns  Ttx instead of Ttt

    shear_relax_factor should be '5' or '0p5'
    """

    variables = {'Ttt':4, 'Ttx':5}

    filename='{}_tfinal_{}.dat'.format(visMUSIC, tfinal)
    foldername="./music_data/shear_relax_%s/%s/" % (shear_relax_factor, testcase)
    MUSICdata=np.loadtxt(foldername + filename, skiprows=1)
    nx,ny = MUSICdata.shape
    xmusic = MUSICdata[:,1] 

    # Get Either the  Ttt component or Ttx component
    Tttmusic = MUSICdata[:,variables[tag]]

    return xmusic, Tttmusic


################################################################################
# Structure holding the parameters for test1 and test2 which will be compared to kinetic theory. This is used  by getDFdata
testcases = { 'test1' : 
                 {'const': 0.12, 'amplitude': 0.48  , 'width': 25.0, 'finaltime': 50.0, }, 
              'test2'  :
                 {'const': 0.06 , 'amplitude': 9.6 , 'width':  25.0, 'finaltime': 50.0, }, 
             }

# Translation between lambda and eta/s
etabys_of_lambdaekt={20:0.180, 10:0.513, 5:1.48}

# List of available lambda
list_lambdaekt=[20, 10, 5]

# Corresponding eta
list_etabys=[0.180, 0.513, 1.48] 


# this returns the final names for the kinetic theory tests and is a helper for getdfdata
def getnames(fourpietabys, gaussian_const, gaussian_amplitude, gaussian_width):
    s='DFAndBDNK/nbys_{}_d_{}_A_{}_w_{}'.format(fourpietabys, gaussian_const, gaussian_amplitude, gaussian_width)
    return './' + s.replace('.', 'd')



def getDFdata(lambdaekt,testcase='test1', tag='Ttt', fourpietabys_in=None):

    """  Returns the density frame and BDNK data for a given run 

    The output is x[:], TttDF[:], TttBDNK[:], TttIdeal[:], info


    The info is an associative array containing the parameters of the run. If tag  can be 'Ttt' , 'Ttx', 'eps', or 'ux'
    """

    fourpietabys = 4*np.pi*etabys_of_lambdaekt[lambdaekt]

    variables = {'Ttt':0, 'Ttx':1, 'eps':2, 'ux':3 }  
    params = testcases[testcase]

    name = getnames(fourpietabys, params['const'], params['amplitude'] , params['width'])
    name0 = getnames(0.0, params['const'], params['amplitude'], params['width'])

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
def freefunction(testcase='test1'):
    """ Returns for the free theory 

        x[:], Ttt[:], Ttx[:]

    """

    x=np.linspace(-75.1,75.1,500)
    delta=testcases[testcase]['const']
    A=testcases[testcase]['amplitude'] 
    w=np.sqrt(testcases[testcase]['width'])
    tc=testcases[testcase]['finaltime']

    analytic=delta+np.sqrt(np.pi)/2*A*w/(2*tc)*(sp.erf((x+tc)/w)-sp.erf((x-tc)/w))
    analyticflux= A*w/(4*tc**2)*( w*( np.exp(-(tc+x)**2/w**2  ) - np.exp(-(tc-x)**2/w**2) ) +   np.sqrt(np.pi)*x*(sp.erf((x+tc)/w)-sp.erf((x-tc)/w))  )

    return  x, analytic, analyticflux


#########################################################################
#def plotIC(case='DF', lambda_case=0):
#    """Plots the initial conditions """ 

#    listetas=[0.180, 0.513, 1.48]
#    listlambda=[20, 10, 5]

#    etas4pi=4*np.pi*np.array(listetas)

#    gaussian_const = 0.12
#    gaussian_amplitude = 0.48
#    gaussian_width = 25.0

#    fig1, ax1 = plt.subplots()
#    ax1.set_xlim(-15, 15)
#    #ax1.set_xticks([-75,-50,-25,0,25,50,75])
#    ax1.set_ylim(-0.1, 0.65)

#    i = lambda_case

#    xfree0, freeTtt0, freeTtx0 = freefunction(0.000001)


#    if case == 'DF':
#        ax1.plot(xfree0, freeTtt0, "C0", linewidth=1.2, label='Density Frame') 
#    else:
#        ax1.plot(xfree0, freeTtt0,  "C0", linewidth=1.2, label='BDNK') 

#    ax1.plot(xfree0, freeTtt0, "C3", linestyle=(0,(3,2)), linewidth=1.2, label='QCD kinetics')

#    # ax1.plot(x, ideal, "k--", linewidth=0.5, label=r"$\eta/s=0$ and $\infty$") 
#    # ax1.plot(xfree, freeTtt, "k--", linewidth=0.5)

#    ax1.legend(frameon=True,loc="lower left", fancybox=False, framealpha=0.8, edgecolor='white')
#    # # Set a text label with the value of eta/s
#    ax1.annotate('initial conditions', (0.85,0.25), xycoords='figure fraction', ha='right',bbox=dict(alpha=0.8,facecolor='white',edgecolor='white'))

#    ax1.annotate(r'$A=0.48\;{\rm GeV}$'+"\n" + r"$\delta=0.12\; {\rm GeV}$" + "\n" + r"$L=5 \;{\rm GeV}^{-1}$", xy=(0.7,0.7), xycoords='figure fraction', linespacing=1.5)

#    ax1.set_xlabel(r'$x$')
#    ax1.set_ylabel(r'$T^{tt}$')
#    fig1.tight_layout() 

#    #name = "{}_{}.pdf".format(case,listlambda[lambda_case])
#    fig1.savefig("test1_ic.pdf")
#    #fig1.savefig(name)


########################################################################
def plotStress(case='DF', lambdaekt=20):
    """Makes a nice comparison between the density frame and kintec theory or BDNK and kinetic theory """


    fig1, ax1 = plt.subplots()
    ax1.set_xlim(-75, 75)
    ax1.set_xticks([-75,-50,-25,0,25,50,75])
    ax1.set_ylim(0.0, 0.50)

    # Get the kinetic theory data
    lambdaekt 
    et = etabys_of_lambdaekt[lambdaekt] 

    xekt, Tttekt  = getEKTdata(lambdaekt)

    # Get density frame, bdnk, and idealhydro
    x, df, bdnk, ideal, data = getDFdata(lambdaekt)

    # Get the MUSIC data
    xmusic, Tttmusic  = getMUSICdata(et)
    # Get Free streaming data
    xfree, freeTtt, freeTtx = freefunction()


    if case == 'DF':
        ax1.plot(x, df, "C0", linewidth=1.2, label='Density Frame') 
    elif case == 'BDNK':
        ax1.plot(x, bdnk, "C0", linewidth=1.2, label='BDNK') 
    else:
        ax1.plot(xmusic, Tttmusic, "C0", linewidth=1.2, label='MUSIC') 

    ax1.plot(xekt, Tttekt, "C3", linestyle=(0,(3,2)), linewidth=1.2, label='QCD kinetics')

    ax1.plot(x, ideal, "k--", linewidth=0.5, label=r"$\eta/s=0$ and $\infty$") 
    ax1.plot(xfree, freeTtt, "k--", linewidth=0.5)

    # ax1.legend(frameon=True,loc="lower left", fancybox=False, framealpha=0.8, edgecolor='white')
    # Set a text label with the value of eta/s
    ax1.annotate(r'$4\pi\eta/s={:.1f}$'.format(4.0*np.pi*et), (0.85,0.25), xycoords='figure fraction', ha='right',bbox=dict(alpha=0.8,facecolor='white',edgecolor='white'))


    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$T^{tt}$')
    fig1.tight_layout() 

    name = "test1_{}_{}.pdf".format(case,lambdaekt)
    fig1.savefig(name)
        
################################################################################

def plotKTPlot1(case='DF'):
    """Makes a nice comparison between the density frame and kintec theory for moderate coupling """

    fig1, ax1 = plt.subplots()
    ax1.set_xlim(-75, 75)
    ax1.set_xticks([-75,-50,-25,0,25,50,75])
    ax1.set_ylim(0.0, 0.32)

    for i in range(0,2):
        # Get density frame, bdnk, and idealhydro
        x, df, bdnk, ideal, data = getDFdata(list_lambdaekt[i])

        # Get the MUSIC data
        xmusic, Tttmusic  = getMUSICdata(list_etabys[i])
        # For the first plot add the label
        if i == 0: 
            if case == 'DF':
                ax1.plot(x, df, "C0", linewidth=1.0, label='density frame') 
            elif case == 'MUSIC':
                ax1.plot(xmusic, Tttmusic, "C6", linewidth=1.0, label='MUSIC') 
        else:
            if case == 'DF':
                ax1.plot(x, df, "C0", linewidth=1.0) 
            elif case == 'MUSIC':
                ax1.plot(xmusic, Tttmusic, "C6", linewidth=1.0) 


    for i in range(0,2):

        # Get the kinetic theory data
        lambdaekt= list_lambdaekt[i]
        et = list_etabys[i]

        xekt, Tttekt  = getEKTdata(lambdaekt)

        # Get Free streaming data
        xfree, freeTtt, freeTtx = freefunction()


        # For the first plot add the label
        if i == 0: 
            ax1.plot(xekt, Tttekt, "C3", linewidth=1.0, linestyle="--", label='QCD kinetics')
        else:
            ax1.plot(xekt, Tttekt, "C2", linewidth=1.0, linestyle="--")

    ax1.plot(x, ideal, "k--", linewidth=0.5, label="ideal hydro") 
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$T^{tt}$')

    # # Get Free streaming data
    # xfree, freeTtt, freeTtx = freefunction()
    # ax1.plot(xfree, freeTtt, "k--", linewidth=0.5, label="free streaming") 

    ax1.annotate(r'$4\pi\eta/s$={:.1f} and ${:.1f}$'.format(4.0*np.pi*list_etabys[0], 4.0*np.pi*list_etabys[1]), (0.85,0.25), xycoords='figure fraction', ha='right',bbox=dict(alpha=0.8,facecolor='white',edgecolor='white'))

    ax1.legend(loc="lower left")
    fig1.tight_layout() 

    fig1.savefig('KTPlot1.pdf')


def plotKTPlot1b(case='DF'):
    """Makes a nice comparison between the density frame and kintec theory for weak coupling """



    fig1, ax1 = plt.subplots()
    ax1.set_xlim(-75, 75)
    ax1.set_xticks([-75,-50,-25,0,25,50,75])
    ax1.set_ylim(0.0, 0.32)

    i= 2
    # Get the kinetic theory data
    lambdaekt = list_lambdaekt[i] 
    xekt, Tttekt  = getEKTdata(lambdaekt)

    # Get density frame, bdnk, and idealhydro
    x, df, bdnk, ideal, data = getDFdata(lambdaekt)

    # Get Free streaming data
    xfree, freeTtt, freeTtx = freefunction()

    # Get the MUSIC data
    xmusic, Tttmusic  = getMUSICdata(list_etabys[i])

    if case == 'DF':
        ax1.plot(x, df, "C0", linewidth=1.0, label='density frame') 
    elif case == 'MUSIC':
        ax1.plot(xmusic, Tttmusic, "C6", linewidth=1.0, label='MUSIC') 

    ax1.plot(xekt, Tttekt, "C7" , linewidth=1.2, linestyle="--", label='QCD kinetics')

    ax1.plot(xfree, freeTtt, "k--", linewidth=0.5, label="free streaming") 

    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$T^{tt}$')

    ax1.annotate(r'$4\pi\eta/s={:.1f}$'.format(4.0*np.pi*list_etabys[2]), (0.85,0.25), xycoords='figure fraction', ha='right',bbox=dict(alpha=0.8,facecolor='white',edgecolor='white'))

    ax1.legend(loc="lower left")

    fig1.savefig('KTPlot1b.pdf')

def plotKTPlot2(case='DF'):
    """Makes a nice comparison between the density frame and kintec theory for moderate coupling """

    fig1, ax1 = plt.subplots()
    ax1.set_xlim(-75, 75)
    ax1.set_xticks([-75,-50,-25,0,25,50,75])
    ax1.set_ylim(0.0, 3.7)

    for i in range(0,2):
        # Get density frame, bdnk, and idealhydro
        x, df, bdnk, ideal, data = getDFdata(list_lambdaekt[i], testcase='test2')

        # Get the MUSIC data
        xmusic, Tttmusic  = getMUSICdata(list_etabys[i], testcase='test2')
        # For the first plot add the label
        if i == 0: 
            if case == 'DF':
                ax1.plot(x, df, "C0", linewidth=1.0, label='density frame') 
            elif case == 'MUSIC':
                ax1.plot(xmusic, Tttmusic, "C6", linewidth=1.0, label='MUSIC') 
        else:
            if case == 'DF':
                ax1.plot(x, df, "C0", linewidth=1.0) 
            elif case == 'MUSIC':
                ax1.plot(xmusic, Tttmusic, "C6", linewidth=1.0) 


    for i in range(0,2):

        # Get the kinetic theory data
        lambdaekt= list_lambdaekt[i]
        et = list_etabys[i]

        xekt, Tttekt  = getEKTdata(lambdaekt, testcase='test2')

        # Get Free streaming data
        xfree, freeTtt, freeTtx = freefunction(testcase='test2')


        # For the first plot add the label
        if i == 0: 
            ax1.plot(xekt, Tttekt, "C3", linewidth=1.0, linestyle="--", label='QCD kinetics')
        else:
            ax1.plot(xekt, Tttekt, "C2", linewidth=1.0, linestyle="--")

    ax1.plot(x, ideal, "k--", linewidth=0.5, label="ideal hydro") 
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$T^{tt}$')

    # # Get Free streaming data
    # xfree, freeTtt, freeTtx = freefunction()
    # ax1.plot(xfree, freeTtt, "k--", linewidth=0.5, label="free streaming") 

    ax1.annotate(r'$4\pi\eta/s$={:.1f} and ${:.1f}$'.format(4.0*np.pi*list_etabys[0], 4.0*np.pi*list_etabys[1]), (0.85,0.80), xycoords='figure fraction', ha='right',bbox=dict(alpha=0.8,facecolor='white',edgecolor='white'))

    ax1.legend(loc="upper left")
    fig1.tight_layout() 

    fig1.savefig('KTPlot2.pdf')


def plotKTPlot2b(case='DF'):
    """Makes a nice comparison between the density frame and kintec theory for weak coupling """

    fig1, ax1 = plt.subplots()
    ax1.set_xlim(-75, 75)
    ax1.set_xticks([-75,-50,-25,0,25,50,75])
    ax1.set_ylim(0.0, 3.7)

    i= 2
    # Get the kinetic theory data
    lambdaekt = list_lambdaekt[i] 
    xekt, Tttekt  = getEKTdata(lambdaekt,testcase='test2')

    # Get density frame, bdnk, and idealhydro
    x, df, bdnk, ideal, data = getDFdata(lambdaekt ,  testcase='test2')

    # Get Free streaming data
    xfree, freeTtt, freeTtx = freefunction(testcase='test2')

    # Get the MUSIC data
    xmusic, Tttmusic  = getMUSICdata(list_etabys[i], testcase='test2')

    if case == 'DF':
        ax1.plot(x, df, "C0", linewidth=1.0, label='density frame') 
    elif case == 'MUSIC':
        ax1.plot(xmusic, Tttmusic, "C6", linewidth=1.0, label='MUSIC') 

    ax1.plot(xekt, Tttekt, "C7" , linewidth=1.2, linestyle="--", label='QCD kinetics')

    ax1.plot(xfree, freeTtt, "k--", linewidth=0.5, label="free streaming") 

    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$T^{tt}$')

    ax1.annotate(r'$4\pi\eta/s={:.1f}$'.format(4.0*np.pi*list_etabys[2]), (0.85,0.80), xycoords='figure fraction', ha='right',bbox=dict(alpha=0.8,facecolor='white',edgecolor='white'))

    ax1.legend(loc="upper left")

    fig1.savefig('KTPlot2b.pdf')

def plotKTOnlyPlot2(case='DF',figname='KineticOnlyT2.pdf'):
    """ Plots kinetic theory results for test1 """


    fig1, ax1 = plt.subplots()
    ax1.set_xlim(-75, 75)
    ax1.set_ylim(0.0, 0.5)
    ax1.set_xticks([-75,-50,-25,0,25,50,75])
    ax1.set_ylim(0.0, 4.0)
    #ax1.set_ylim(0.0, 0.35)

    #colors = ['C3',mcolors.CSS4_COLORS['peachpuff'],'C2']
    colors = ['C3', 'C2','C7']
    #colors = ['C3',mcolors.TABLEAU_COLORS['tab:pink'],'C2']
    #colors = ['C3',mcolors.BASE_COLORS['m'],'C2']


    labels=[]
    for i  in range(0,3):
        labels.append(r'$4\pi\eta/s={:.1f}$'.format(4.0*np.pi*list_etabys[i]))


    for i in range(0,3):

        # Get the kinetic theory data
        et = list_etabys[i] 
        lambdaekt = list_lambdaekt[i]
        xekt, Tttekt = getEKTdata(lambdaekt, testcase='test2')

        # Get density frame, bdnk, and idealhydro
        x, df, bdnk, ideal, data = getDFdata(lambdaekt, testcase='test2')

        # Get Free streaming data
        xfree, freeTtt, freeTtx = freefunction(
            testcase='test2')

        #ax1.plot(xekt, Tttekt,'o', markersize=2.0, markerfacecolor='none', markeredgecolor=colors[i], markeredgewidth=0.5, linewidth=0.5) 

        #ax1.plot(x, df,color='C0', linewidth=1.2)

        ax1.plot(xekt, Tttekt,"--", color=colors[i], linewidth=1.0) 


        if i == 0:
            ax1.plot(x, ideal, "k--", linewidth=0.5, label=r"ideal hydro") 
        #
        # ax1.plot(xekt, Tttekt, color=colors[i], linewidth=1.2, linestyle="--", label=labels[i])

    ax1.plot(xfree, freeTtt, "k:", linewidth=0.7, label=r"free stream") 
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$T^{tt}$')
    ax1.legend(loc="upper center",fontsize=8)

    #ax1.annotate("QCD kinetics\ntest 1", (0.87,0.8), ha='right', xycoords='figure fraction')
    fig1.savefig(figname)

def plotKTOnlyPlot1(case='DF',figname='KineticOnlyT1.pdf'):
    """ Plots kinetic theory results for test1 """


    fig1, ax1 = plt.subplots()
    ax1.set_xlim(-75, 75)
    ax1.set_ylim(0.0, 0.5)
    ax1.set_xticks([-75,-50,-25,0,25,50,75])
    ax1.set_ylim(0.0, 0.35)

    #colors = ['C3',mcolors.CSS4_COLORS['peachpuff'],'C2']
    colors = ['C3', 'C2','C7']
    #colors = ['C3',mcolors.TABLEAU_COLORS['tab:pink'],'C2']
    #colors = ['C3',mcolors.BASE_COLORS['m'],'C2']


    labels=[]
    for i  in range(0,3):
        labels.append(r'$4\pi\eta/s={:.1f}$'.format(4.0*np.pi*list_etabys[i]))


    for i in range(0,3):

        # Get the kinetic theory data
        et = list_etabys[i] 
        lambdaekt = list_lambdaekt[i]
        xekt, Tttekt = getEKTdata(lambdaekt, testcase='test1')

        # Get density frame, bdnk, and idealhydro
        x, df, bdnk, ideal, data = getDFdata(lambdaekt, testcase='test1')

        # Get Free streaming data
        xfree, freeTtt, freeTtx = freefunction()

        #ax1.plot(xekt, Tttekt,'o', markersize=2.0, markerfacecolor='none', markeredgecolor=colors[i], markeredgewidth=0.5, linewidth=0.5) 

        #ax1.plot(x, df,color='C0', linewidth=1.2)

        ax1.plot(xekt, Tttekt,"--", color=colors[i], linewidth=1.0) 


        if i == 0:
            ax1.plot(x, ideal, "k--", linewidth=0.5, label=r"ideal hydro") 
        #
        # ax1.plot(xekt, Tttekt, color=colors[i], linewidth=1.2, linestyle="--", label=labels[i])

    ax1.plot(xfree, freeTtt, "k:", linewidth=0.8, label=r"free stream") 
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$T^{tt}$')
    ax1.legend(loc="upper center",fontsize=8)

    #ax1.annotate("QCD kinetics\ntest 1", (0.87,0.8), ha='right', xycoords='figure fraction')
    fig1.savefig(figname)

plotKTOnlyPlot2()
#plotKTOnlyPlot1()
# # plotIC()
# plotStress(lambdaekt=20)
# plotStress(lambdaekt=10)
# plotStress(lambdaekt=5)
# plotStress(case='DF', lambdaekt=20)
# plotStress(case='DF', lambdaekt=10)
# plotStress(case='DF', lambdaekt=5)
# plotStress(case='BDNK', lambdaekt=20)
# plotStress(case='BDNK', lambdaekt=10)
# plotStress(case='BDNK', lambdaekt=5)
# plotStress(case='MUSIC', lambdaekt=20)
# plotStress(case='MUSIC', lambdaekt=10)
# plotStress(case='MUSIC', lambdaekt=5)

#plotKTPlot1(case='MUSIC')
#plotKTPlot1b(case='DF')

plotKTPlot2(case='DF')
plotKTPlot2b(case='DF')
#plotKTPlot1b(case='DF')
