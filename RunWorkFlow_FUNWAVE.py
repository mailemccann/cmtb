# -*- coding: utf-8 -*-
import multiprocessing
import matplotlib
# matplotlib.use('Agg')
import os, getopt, sys, shutil, glob, logging, yaml, time, pickle
import datetime as DT
from subprocess import check_output
import numpy as np
from frontback import frontBackFUNWAVE
from getdatatestbed import getDataFRF
from testbedutils import fileHandling
import pickle

def Master_FUNWAVE_run(inputDict):
    """This function will run FUNWAVE with any version prefix given start, end, and timestep

    Args:
      inputDict: a dictionary that is read from the input yaml

    Keyword Args:
        modelSettings:
            'version_prefix'
            'ensembleNumber': values for ensembles to run (default = np.arange(1))
        hostfile: hostfile (default = '/home/number/cmtb/hostfile-IB_funwave')

    Returns:
      None

    """
    ## unpack Dictionary
    version_prefix = inputDict['modelSettings'].get('version_prefix', 'base').lower()
    grid = inputDict['modelSettings'].get('grid').lower()
    endTime = inputDict['endTime']
    startTime = inputDict['startTime']
    simulationDuration = inputDict['simulationDuration']
    workingDir = inputDict.get('workingDirectory', 'data')
    generateFlag = inputDict['generateFlag']
    runFlag = inputDict['runFlag']
    pbsFlag = inputDict['pbsFlag']
    analyzeFlag = inputDict['analyzeFlag']
    plotFlag = inputDict['plotFlag']
    model = inputDict.get('modelName', 'FUNWAVE').lower()
    inputDict['path_prefix'] = os.path.join(workingDir, model, version_prefix)

    path_prefix = inputDict['path_prefix']
    ensembleNumber = inputDict['modelSettings'].get('ensembleNumber', np.arange(0,1))
    hostfile = inputDict.get('hostFile', '/home/number/cmtb/hostfile-IB_funwave')
    # ______________________ Logging  ____________________________
    # auto generated Log file using start_end timeSegment
    LOG_FILENAME = fileHandling.logFileLogic(outDataBase=path_prefix, version_prefix=version_prefix, startTime=startTime,
                                             endTime=endTime, log=False)
    # ____________________________________________________________
    # establishing the resolution of the input datetime
    projectEnd = DT.datetime.strptime(endTime, '%Y-%m-%dT%H:%M:%SZ')
    projectStart = DT.datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%SZ')
    # This is the portion that creates a list of simulation end times
    # make List of Datestring items, for simulations
    dateStartList, dateStringList, projectStart, projectEnd = fileHandling.createTimeInfo(projectStart, projectEnd,
                                                                              simulationDuration=simulationDuration)
    errors, errorDates = [],[]
    curdir = os.getcwd()
    # ______________________________decide process and run _____________________________
    fileHandling.displayStartInfo(projectStart, projectEnd, version_prefix, LOG_FILENAME, model)
    fileHandling.checkVersionPrefix(model, inputDict)
    # ______________________________Get data to run model  _____________________________
    # begin model data gathering
    go = getDataFRF.getObs(projectStart, projectEnd, server='FRF')                  # initialize get observation class
    gdTB = getDataFRF.getDataTestBed(projectStart, projectEnd)        # for bathy data gathering


    if version_prefix.lower() in ['freq', 'freq-ee']:
        #load specific date/time of interest
        with open('grids/FUNWAVE/bathyPickle_{}.pickle'.format(projectStart.strftime("%Y-%m-%d")), 'rb') as fid:
            bathy = pickle.load(fid)
        if version_prefix.lower() in ['freq']:  # specifically focused on df
            with open('grids/FUNWAVE/phases.pickle', 'rb') as fid:
                phases = pickle.load(fid)
            #freqList = inputDict['modelSettings']['freqList']
            freqList = inputDict['modelSettings']['freqList']
            ensembleNumber = [int(i) for i in ensembleNumber.split(',')]
            # check to make sure keys got into pickle appropriately
            for dfKey in freqList:
                if any(phase.startswith(dfKey)for phase in phases.keys()):
                    print('  {} key not in pickle'.format(dfKey))
                for i in ensembleNumber:
                    if len(phases['phase_{}_{}'.format(dfKey, i)]) == 0:
                        print('failed phase_{}_{}'.format(dfKey, i))
        else:
            freqList = ['a']
            ensembleNumber = ['1']
    else:
        bathy = gdTB.getBathyIntegratedTransect(method=1, ybounds=[940, 950])
        freqList = ['a']
        ensembleNumber = ['1']
    if generateFlag is True:
        rawspec = go.getWaveSpec(gaugenumber='8m-array')
        rawWL = go.getWL()

    # _____________________________ RUN LOOP ___________________________________________
    for dfKey in freqList:                      # loop through frequency members
        for enMb in ensembleNumber:           # loop through ensemble members
            if version_prefix.lower() in ['freq']:
                inputDict['phases'] = phases['phase_{}_{}'.format(dfKey, enMb)]
                assert len(inputDict['phases']) == len(phases['phase_{}_freq'.format(dfKey)]), "some how picked the wrong phase"
            try:
                dateString = os.path.join(dateStartList[0].strftime('%Y-%m-%dT%H%M%SZ'),'phase_{}_{}'.format(dfKey, enMb)) #'phase_{}_{}'.format(dfKey, enMb) #projectStart.strftime("%Y%m%dT%H%M%SZ")
                fileHandling.makeCMTBfileStructure(path_prefix=path_prefix, date_str=dateString)
                datadir = os.path.join(path_prefix, dateString)  # moving to the new simulation's folder
                pickleSaveFname = os.path.join(datadir, 'phase_{}_{}'.format(dfKey, enMb)+'_io.pickle')

                if generateFlag == True:
                    # assigning min/max frequency bands with resolution of df key
                    if version_prefix in ['freq']: inputDict['nf'] = len(np.arange(0.04, 0.3, float(dfKey[3:])))
                    fIO = frontBackFUNWAVE.FunwaveSimSetup(dateString, rawWL, rawspec, bathy, inputDict=inputDict)
                    with open(pickleSaveFname, 'wb') as fid:
                        pickle.dump(fIO, fid, protocol=pickle.HIGHEST_PROTOCOL)

                if runFlag == True:        # run model
                    os.chdir(datadir)      # changing locations to where input files should be made
                    dt = time.time()

                    count = multiprocessing.cpu_count()
                    #if count < fIO.nprocess:
                    #    raise EnvironmentError('simulation is calling for more cores than are avialable, please check logic')
                        # this could be the logic associated with generating px,py (core counts requested) this could be
                        # with the logic asking how many cores are available

                    print('Running Simulation with {} processors'.format(fIO.nprocess))
                    if pbsFlag == True:
                        executionString = "qsub submit_script.pbs"
                    else:
                        executionString = "mpiexec -n {} -f {} {} input.txt".format(int(fIO.nprocess), hostfile,
                                                                    os.path.join(curdir, inputDict['modelExecutable']))

                    print(executionString)

                    _ = check_output(executionString, shell=True)
                    fIO.simulationWallTime = time.time() - dt
                    print('Simulation took {:.1} hours'.format(fIO.simulationWallTime/60))

                    os.chdir(curdir)
                    with open(pickleSaveFname, 'wb') as fid:
                        pickle.dump(fIO, fid, protocol=pickle.HIGHEST_PROTOCOL)
    
                else:   # assume there is a saved pickle of input/output that was generated before
                    with open(pickleSaveFname, 'rb') as fid: ## DEBUG Gaby: this is so i can use hpc simulaton
                        fIO = pickle.load(fid)

                if analyzeFlag == True:
                    print('**\nBegin Analyze Script %s ' % DT.datetime.now())
                    fIO.path_prefix = os.path.join(workingDir, model, version_prefix, dateString)
                    frontBackFUNWAVE.FunwaveAnalyze(dateString, inputDict, fIO)

                if plotFlag is True and DT.date.today() == projectEnd:
                    print('  TODO tar simulation files after generating netCDF')
                    # move files
                    moveFnames = glob.glob(curdir + 'cmtb*.png')
                    moveFnames.extend(glob.glob(curdir + 'cmtb*.gif'))
                    for file in moveFnames:
                        shutil.move(file,  '/mnt/gaia/cmtb')
                        print('moved %s ' % file)
                print('------------------Model Run: SUCCESSS-----------------------------------------')

            except Exception as e:
                print('______________<< ERROR >> HAPPENED IN THIS STEP ')
                print(e)
                logging.exception('\nERROR FOUND', exc_info=True)
                os.chdir(curdir)  # change back to main directory (no matter where the simulation failed)


if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "h", ["help"])
    print('___________________\n________________\n___________________\n________________\n___________________\n________________\n')
    print('USACE FRF Coastal Model Test Bed')

    # we are no longer allowing a default yaml file.
    # It will throw and error and tell the user where to go look for the example yaml
    try:
        # assume the user gave the path
        yamlLoc = args[0]

        with open(os.path.join(yamlLoc), 'r') as f:
            inputDict = yaml.safe_load(f)
        if os.path.exists('.cmtbSettings'):
            with open('.cmtbSettings', 'r') as fid:
                a = yaml.safe_load(fid)
            inputDict.update(a)
    except:
        raise IOError('Input YAML file required.  See yaml_files/TestBedExampleInputs/FUNWAVE_Input_example for '
                      'example yaml file.')
    Master_FUNWAVE_run(inputDict=inputDict)
