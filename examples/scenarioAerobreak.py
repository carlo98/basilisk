
import os

import matplotlib.pyplot as plt
import numpy as np
# The path to the location of Basilisk
# Used to get the location of supporting data.
from Basilisk import __path__
from Basilisk.simulation import facetDragDynamicEffector
# import simulation related support
from Basilisk.simulation import spacecraft
from Basilisk.simulation import tabularAtmosphere, simpleNav
# import general simulation support files
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.utilities import orbitalMotion
from Basilisk.utilities import simIncludeGravBody
from Basilisk.utilities import unitTestSupport
from Basilisk.utilities import vizSupport
from Basilisk.utilities.readAtmTable import readAtmTable
from Basilisk.simulation import hingedRigidBodyStateEffector
import math

bskPath = __path__[0]
fileName = os.path.basename(os.path.splitext(__file__)[0])

# THIS DID NOT WORK 
def sph2rv(xxsph):
    """
    NOTE: this function assumes inertial and planet-fixed frames are aligned
    at this time
    """
    
    r = xxsph[0]
    lon = xxsph[1]
    lat = xxsph[2]
    u = xxsph[3]
    gam = xxsph[4]
    hda = xxsph[5]
    
    NI = np.eye(3)
    IE = np.array([[np.cos(lat) * np.cos(lon), -np.sin(lon), -np.sin(lat) * np.cos(lon)],
                   [np.cos(lat) * np.sin(lon), np.cos(lon), -np.sin(lat) * np.sin(lon)],
                   [np.sin(lat), 0, np.cos(lat)]])
    ES = np.array([[np.cos(gam), 0, np.sin(gam)],
                   [-np.sin(gam) * np.sin(hda), np.cos(hda), np.cos(gam) * np.sin(hda)],
                   [-np.sin(gam) * np.cos(hda), -np.sin(hda), np.cos(gam) * np.cos(hda)]])
    
    e1_E = np.array([1,0,0])
    rvec_N = (r * NI @ IE) @ e1_E
    
    s3_S = np.array([0,0,1])
    uvec_N = u * ( NI @ IE @ ES) @ s3_S
    
    return rvec_N, uvec_N


def run(show_plots, planetCase, deorbitAlt=90):
    """
    The scenarios can be run with the followings setups parameters:

    Args:
        show_plots (bool): Determines if the script should display plots
        planetCase (string): Specify if a `Mars` or `Earth` arrival is simulated

    """

    # Create simulation variable names
    simTaskName = "simTask"
    simProcessName = "simProcess"

    #  Create a sim module as an empty container
    scSim = SimulationBaseClass.SimBaseClass()

    #
    #  create the simulation process
    #
    dynProcess = scSim.CreateNewProcess(simProcessName)

    # create the dynamics task and specify the integration update time
    simulationTimeStep = macros.sec2nano(0.1)
    dynProcess.addTask(scSim.CreateNewTask(simTaskName, simulationTimeStep))

    # Construct algorithm and associated C++ container
    # change module to tabAtmo
    tabAtmo = tabularAtmosphere.TabularAtmosphere()   # update with current values
    tabAtmo.ModelTag = "tabularAtmosphere"            # update python name of test module
    atmoTaskName = "atmosphere"
    
    # define constants & load data
    if planetCase == 'Earth':
        r_eq = 6378136.6
        dataFileName = bskPath + '/supportData/AtmosphereData/EarthGRAMNominal.txt'
        altList, rhoList, tempList = readAtmTable(dataFileName, 'EarthGRAM')
    else:
        r_eq = 3397.2 * 1000
        dataFileName = bskPath + '/supportData/AtmosphereData/MarsGRAMNominal.txt'
        altList, rhoList, tempList = readAtmTable(dataFileName, 'MarsGRAM')
        
    # assign constants & ref. data to module
    tabAtmo.planetRadius = r_eq
    tabAtmo.altList = tabularAtmosphere.DoubleVector(altList)    
    tabAtmo.rhoList = tabularAtmosphere.DoubleVector(rhoList)
    tabAtmo.tempList = tabularAtmosphere.DoubleVector(tempList)

    # Drag Effector for sc1 
    drag1 = facetDragDynamicEffector.FacetDragDynamicEffector()
    drag1.ModelTag = "FacetDrag1"
    dragEffectorTaskName1 = "drag1" #should there be one task name per drag1 and drag 2?
    # drag1.setDensityMessage(atmoModule.envOutMsgs[0]) # docs say to do this but scenarioDragRendezvous does not do it

    scAreas = 10.0
    scCoeff = 2.0
    
    B_normals = [
        np.array([ 1, 0, 0]), 
        np.array([-1, 0, 0]), 
        np.array([ 0, 1, 0]), 
        np.array([ 0, -1, 0]), 
        np.array([ 0, 0, 1]), 
        np.array([ 0, 0, -1]) 
    ]

    B_locations = [
        np.array([ 0.2, 0.0, 0.0]),
        np.array([-0.2, 0.0, 0.0]),
        np.array([ 0.0, 0.2, 0.0]),
        np.array([ 0.0, -0.2, 0.0]),
        np.array([ 0.0, 0.0, 0.2]),
        np.array([ 0.0, 0.0, -0.2])
    ]
    
    for ind in range(0,len(B_normals)):
        drag1.addFacet(scAreas, scCoeff, B_normals[ind], B_locations[ind])

    # Drag Effectors for sc2
    drag2 = facetDragDynamicEffector.FacetDragDynamicEffector()
    drag2.ModelTag = "FacetDrag2"
    dragEffectorTaskName2 = "drag2"
    # drag2.setDensityMessage(atmoModule.envOutMsgs[0]) # docs say to do this but scenarioDragRendezvous does not do it

    # sc2 has the same hub facets as sc1:
    for ind in range(0,len(B_normals)):
        drag2.addFacet(scAreas, scCoeff, B_normals[ind], B_locations[ind])

    drag3 = facetDragDynamicEffector.FacetDragDynamicEffector()
    drag3.ModelTag = "FacetDrag3"
    dragEffectorTaskName3 = "panelDrag3"

    drag4 = facetDragDynamicEffector.FacetDragDynamicEffector()
    drag4.ModelTag = "FacetDrag4"
    dragEffectorTaskName4 = "panelDrag4"
    
    panelArea = 25    # m^2 ???
    panelOffset = 0.3  # Distance from hub COM
    panelCd = 5

    panel_normals = [
        np.array([0, 0,  1]), 
        np.array([0, 0, -1]) # may need to add other panel to this
    ]

    panel_locations = [
        np.array([0.0, 0.0, panelOffset]),
        np.array([0.0, 0.0, panelOffset])
    ]

    for i in range(2):
        drag3.addFacet(panelArea, panelCd, panel_normals[i], panel_locations[i])

    for i in range(2):
        drag4.addFacet(panelArea, panelCd, panel_normals[i], panel_locations[i])
    
    for i in range(2):
        drag1.addFacet(panelArea, panelCd, panel_normals[i], panel_locations[i])

    dynProcess.addTask(scSim.CreateNewTask(atmoTaskName, simulationTimeStep))
    
    dynProcess.addTask(scSim.CreateNewTask(dragEffectorTaskName1, simulationTimeStep))
    dynProcess.addTask(scSim.CreateNewTask(dragEffectorTaskName2, simulationTimeStep))
    dynProcess.addTask(scSim.CreateNewTask(dragEffectorTaskName3, simulationTimeStep))
    dynProcess.addTask(scSim.CreateNewTask(dragEffectorTaskName4, simulationTimeStep))
    
    scSim.AddModelToTask(atmoTaskName, tabAtmo)

    #
    #   setup the simulation tasks/objects
    #

    m_sc = 2530.0    # kg

    # initialize spacecraft object and set properties
    scObject1 = spacecraft.Spacecraft() # panel fixed over time from POV of drag 
    scObject1.ModelTag = "spacecraftBody1"
    scObject1.hub.mHub = m_sc
    tabAtmo.addSpacecraftToModel(scObject1.scStateOutMsg)

    scObject2 = spacecraft.Spacecraft() # one facited drag to hub and another attached to panel (2 - one for each side)
    scObject2.ModelTag = "spacecraftBody2"
    scObject2.hub.mHub = m_sc
    tabAtmo.addSpacecraftToModel(scObject2.scStateOutMsg)
    
    simpleNavObj = simpleNav.SimpleNav()
    scSim.AddModelToTask(simTaskName, simpleNavObj)
    simpleNavObj.scStateInMsg.subscribeTo(scObject1.scStateOutMsg)
    simpleNavObj.scStateInMsg.subscribeTo(scObject2.scStateOutMsg)

    # Panel 1 is not the parent to the drag effector 
    panel1 = hingedRigidBodyStateEffector.HingedRigidBodyStateEffector() 
    panel1.ModelTag = "panel1"

    # Panel 2 is the parent to the drag effector (and so is the hub)
    panel2 = hingedRigidBodyStateEffector.HingedRigidBodyStateEffector()
    panel2.ModelTag = "panel2"

    panel1.mass = 100
    panel1.IPntS_S = [[100.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 50.0]]
    panel1.d = 1.5
    panel1.k = 200
    panel1.c = 20  # c is the rotational damping coefficient for the hinge, which is modeled as a spring.
    panel1.r_HB_B = [[0.5], [0.0], [-1.0]] # maybe make it 1 
    panel1.dcm_HB = [[-1, 0, 0.0], [0.0, -1, 0.0], [0.0, 0.0, 1]]
    panel1.thetaInit = 0.0
    panel1.thetaDotInit = 0.0

    panel2.mass = 100
    panel2.IPntS_S = [[100.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 50.0]]
    panel2.d = 1.5
    panel2.k = 200
    panel2.c = 20  # c is the rotational damping coefficient for the hinge, which is modeled as a spring.
    panel2.r_HB_B = [[0.5], [0.0], [-1.0]] 
    panel2.dcm_HB = [[-1, 0, 0.0], [0.0, -1, 0.0], [0.0, 0.0, 1]]
    panel2.thetaInit = 0.0
    panel2.thetaDotInit = 0.0

    # # Symmetrically opposite panels for each spacecraft
    panel3 = hingedRigidBodyStateEffector.HingedRigidBodyStateEffector() 
    panel3.ModelTag = "panel3"

    panel4 = hingedRigidBodyStateEffector.HingedRigidBodyStateEffector() 
    panel4.ModelTag = "panel4"

    panel3.mass = 100
    panel3.IPntS_S = [[100.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 50.0]]
    panel3.d = -1.5
    panel3.k = 200
    panel3.c = 20  # c is the rotational damping coefficient for the hinge, which is modeled as a spring.
    panel3.r_HB_B = [[-0.5], [0.0], [-1.0]]
    panel3.dcm_HB = [[-1, 0, 0.0], [0.0, -1, 0.0], [0.0, 0.0, 1]]
    panel3.thetaInit = 0.0
    panel3.thetaDotInit = 0.0

    panel4.mass = 100
    panel4.IPntS_S = [[100.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 50.0]]
    panel4.d = -1.5
    panel4.k = 200
    panel4.c = 20  # c is the rotational damping coefficient for the hinge, which is modeled as a spring.
    panel4.r_HB_B = [[-0.5], [0.0], [-1.0]] 
    panel4.dcm_HB = [[-1, 0, 0.0], [0.0, -1, 0.0], [0.0, 0.0, 1]]
    panel4.thetaInit = 0.0
    panel4.thetaDotInit = 0.0

    #########
    # Add panels to spaceCraft
    #########

    # in order to affect dynamics
    scObject1.addStateEffector(panel1)
    scObject2.addStateEffector(panel2)
    scObject1.addStateEffector(panel3)
    scObject2.addStateEffector(panel4)

    scSim.AddModelToTask(simTaskName, scObject1)
    scSim.AddModelToTask(simTaskName, scObject2)

    # in order to track messages
    scSim.AddModelToTask(simTaskName, panel1) 
    scSim.AddModelToTask(simTaskName, panel2)
    scSim.AddModelToTask(simTaskName, panel3)
    scSim.AddModelToTask(simTaskName, panel4)
    
    print("in python attaching to hub 1!!!!")
    # Attach drag to hub only (previous method)
    scObject1.addDynamicEffector(drag1) # remember later to add the other panel facets to the drag!!!
    scSim.AddModelToTask(dragEffectorTaskName1, drag1)
    drag1.atmoDensInMsg.subscribeTo(tabAtmo.envOutMsgs[0])
    
    # Attach drag to hub and panel (new/more accurate model)
    print("in python attaching to hub 2!!!!")
    scObject2.addDynamicEffector(drag2)
    scSim.AddModelToTask(dragEffectorTaskName2, drag2)
    drag2.atmoDensInMsg.subscribeTo(tabAtmo.envOutMsgs[0])

    print("in python attaching to panel!!!!!!")
    panel2.addDynamicEffector(drag3)  
    scSim.AddModelToTask(dragEffectorTaskName3, drag3)
    drag3.atmoDensInMsg.subscribeTo(tabAtmo.envOutMsgs[0])    

    panel4.addDynamicEffector(drag4)  
    scSim.AddModelToTask(dragEffectorTaskName4, drag4)
    drag4.atmoDensInMsg.subscribeTo(tabAtmo.envOutMsgs[0])    

    # set the simulation time
    if planetCase == 'Earth':
        simulationTime = macros.sec2nano(300)
    else:
        simulationTime = macros.sec2nano(400)

    # Setup Gravity Body
    gravFactory = simIncludeGravBody.gravBodyFactory()
    planet = gravFactory.createBody(planetCase)
    planet.isCentralBody = True  # ensure this is the central gravitational body

    # Attach gravity model to spacecraft
    gravFactory.addBodiesTo(scObject1)
    gravFactory.addBodiesTo(scObject2)

    if planetCase == 'Earth':
        r = 6503 * 1000.
        u = 11.2 * 1000
        gam = -5.15 * macros.D2R
    else:
        r = (3397.2 + 125.) * 1000
        u = 6 * 1000
        gam = -10 * macros.D2R
    lon = 0
    lat = 0
    hda = np.pi/2
    xxsph = [r,lon,lat,u,gam,hda]
    rN, vN = sph2rv(xxsph)
    
    scObject1.hub.r_CN_NInit = rN  # m - r_CN_N
    scObject1.hub.v_CN_NInit = vN  # m - v_CN_N
    scObject1.hub.sigma_BNInit = [[math.tan(-90. / 4. * macros.D2R)], [0.0], [0.0]]  # sigma_BN_B
    scObject1.hub.omega_BN_BInit = [[0.0], [0.0], [0.0]]  # rad/s - omega_BN_B

    scObject2.hub.r_CN_NInit = rN  # m - r_CN_N
    scObject2.hub.v_CN_NInit = vN  # m - v_CN_N
    scObject2.hub.sigma_BNInit = [[math.tan(-90. / 4. * macros.D2R)], [0.0], [0.0]]  # sigma_BN_B
    scObject2.hub.omega_BN_BInit = [[0.0], [0.0], [0.0]]  # rad/s - omega_BN_B

    #
    #   Setup data logging before the simulation is initialized
    #

    dataLog1 = scObject1.scStateOutMsg.recorder()
    dataLog2 = scObject2.scStateOutMsg.recorder()
    p1Log = panel1.hingedRigidBodyOutMsg.recorder()
    p2Log = panel2.hingedRigidBodyOutMsg.recorder()
    p3Log = panel3.hingedRigidBodyOutMsg.recorder()
    p4Log = panel4.hingedRigidBodyOutMsg.recorder()


    scSim.AddModelToTask(simTaskName, dataLog1)
    scSim.AddModelToTask(simTaskName, dataLog2)
    scSim.AddModelToTask(simTaskName, p1Log)
    scSim.AddModelToTask(simTaskName, p2Log)
    scSim.AddModelToTask(simTaskName, p3Log)
    scSim.AddModelToTask(simTaskName, p4Log)

    dataNewAtmoLog = tabAtmo.envOutMsgs[0].recorder()
    scSim.AddModelToTask(simTaskName, dataNewAtmoLog)

    # Event to terminate the simulation
    scSim.createNewEvent(
        "Deorbited",
        simulationTimeStep,
        True,
        conditionFunction=lambda self: (
            np.linalg.norm(scObject1.scStateOutMsg.read().r_BN_N)
            < planet.radEquator + 1000 * deorbitAlt
        or 
            np.linalg.norm(scObject2.scStateOutMsg.read().r_BN_N)
            < planet.radEquator + 1000 * deorbitAlt
        ),
        terminal=True,
    )

    scBodyList = [
        scObject1,
        scObject2,
        ["panel1", panel1.hingedRigidBodyConfigLogOutMsg],
        ["panel2", panel2.hingedRigidBodyConfigLogOutMsg],
        ["panel3", panel3.hingedRigidBodyConfigLogOutMsg],
        ["panel4", panel4.hingedRigidBodyConfigLogOutMsg],
    ]

    # if this scenario is to interface with the BSK Viz, uncomment the following line
    viz = vizSupport.enableUnityVisualization(scSim, simTaskName, scBodyList
                                              , saveFile=fileName
                                              )
    # Spacecraft 1 hub
    vizSupport.createCustomModel(viz,
                                    simBodiesToModify=[scObject1.ModelTag],
                                    modelPath="CUBE",
                                    color=vizSupport.toRGBA255("red"),
                                    scale=[1, 2, 3])  # [width, length, height] in meters
    # Spacecraft 2 hub
    vizSupport.createCustomModel(viz,
                                 simBodiesToModify=[scObject2.ModelTag],
                                 modelPath="CUBE",
                                 color=vizSupport.toRGBA255("blue"),
                                 scale=[1, 2, 3])  # [width, length, height] in meters

    # Panel 1 on spacecraft 1
    vizSupport.createCustomModel(viz,
                                 simBodiesToModify=["panel1"],
                                 modelPath="CUBE",
                                 scale=[3, 1, 0.1]) # [width, length, height] in meters
    
    # Panel 3 on spacecraft 1
    vizSupport.createCustomModel(viz,
                                 simBodiesToModify=["panel3"],
                                 modelPath="CUBE",
                                 scale=[3, 1, 0.1]) # [width, length, height] in meters
    
    # Panel 2 on spacecraft 2
    vizSupport.createCustomModel(viz,
                                 simBodiesToModify=["panel2"],
                                 modelPath="CUBE",
                                 color=vizSupport.toRGBA255("gold"),
                                 scale=[3, 1, 0.1])  # [width, length, height] in meters
    
    # Panel 4 on spacecraft 2
    vizSupport.createCustomModel(viz,
                                 simBodiesToModify=["panel4"],
                                 modelPath="CUBE",
                                 color=vizSupport.toRGBA255("gold"),
                                 scale=[3, 1, 0.1]) # [width, length, height] in meters
    
    #
    #   initialize Simulation
    #
    scSim.InitializeSimulation()

    #
    #   configure a simulation stop time and execute the simulation run
    #
    scSim.ConfigureStopTime(simulationTime)
    scSim.ExecuteSimulation()

    #
    #   retrieve the logged data
    #
    posData1 = dataLog1.r_BN_N
    velData1 = dataLog1.v_BN_N
    posData2 = dataLog1.r_BN_N
    velData2 = dataLog1.v_BN_N

    np.set_printoptions(precision=16)

    figureList = {}
    plt.close("all")  # clears out plots from earlier test runs

    # draw the inertial position vector components
    plt.figure(1)
    fig = plt.gcf()
    ax = fig.gca()
    ax.ticklabel_format(useOffset=False, style='plain')
    for idx in range(0,3):
        plt.plot(dataLog1.times()*macros.NANO2MIN, posData1[:, idx]/1000.,
                 color=unitTestSupport.getLineColor(idx,3),
                 label='Drag on hub only: $r_{BN,'+str(idx)+'}$')
    for idx in range(0,3):
        plt.plot(dataLog1.times()*macros.NANO2MIN, posData1[:, idx]/1000.,
                 color=unitTestSupport.getLineColor(idx,3),
                 label='Drag on hub and panel: $r_{BN,'+str(idx)+'}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel('Inertial Position [km]')

    r1 = np.linalg.norm(posData1, axis=1)
    v1 = np.linalg.norm(velData1, axis=1)

    r2 = np.linalg.norm(posData2, axis=1)
    v2 = np.linalg.norm(velData2, axis=1)

    plt.figure(2)
    fig = plt.gcf()
    ax = fig.gca()
    plt.plot(v1/1e3, (r1-r_eq)/1e3, label="Drag on hub only")
    plt.plot(v2/1e3, (r2-r_eq)/1e3, label="Drag on hub and panel")
    plt.xlabel('velocity [km/s]')
    plt.ylabel('altitude [km]')
    plt.grid()
    pltName = fileName + "4" + planetCase
    figureList[pltName] = plt.figure(2)

    plt.figure(3)
    fig = plt.gcf()
    ax = fig.gca()
    plt.plot(dataLog1.times()*macros.NANO2MIN, (r1-r_eq)/1e3, label="Drag on hub only")
    plt.plot(dataLog2.times()*macros.NANO2MIN, (r2-r_eq)/1e3, label="Drag on hub and panel")
    plt.xlabel('time [min]')
    plt.ylabel('altitude [km]')
    plt.grid()
    pltName = fileName + "5" + planetCase
    figureList[pltName] = plt.figure(3)

    if show_plots:
        plt.show()
        plt.close("all")

    return figureList

    # close the plots being saved off to avoid over-writing old and new figures
if __name__ == '__main__':
    run(True, 'Earth', deorbitAlt=90)      # planet arrival case, can be Earth or Mars
    