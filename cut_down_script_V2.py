from tqdm import tqdm
from skimage import (io as skio,measure)
from PyQt5 import QtWidgets
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import statistics


def main(fluofilename, masksfilename, manualchannels):
    """
    Main function - should use just a few functions, offloading specific
    processing to called functions which are defined later.
    fluofilename should be the full fluorescent image tiff stack;
    masksfilename should be the mask image tiff stack produced by animal tracking Ilastik module.
    """

    ''' Prepare the data'''

    [fluoimage,Imasks,Wells] = prep_data(fluofilename, masksfilename, manualchannels)

    #Sets up an empty dictionary which will contain all data from this dataset.
    Data = {}
    
    '''Set up the loop that will call subfunctions'''
    #This loop  iterates once for each frame of the tiff stack.
    for index in tqdm(range(fluoimage.shape[0])):
        Data = process_frame(fluoimage[index], Imasks[index], index, Wells[index], Data)

    return Data

def prep_data(fluofilename,masksfilename,manualchannels):
    '''load the data and set its pixels to binary'''
    # Load data from their tiffs
    fluoimage = skio.imread(fluofilename)
    Imasks = skio.imread(masksfilename)
    Wells = skio.imread(manualchannels)
    if Wells.shape[0] != Imasks.shape[0] != fluoimage.shape[0]:
        raise NameError("Error: Brightfield, Fluorescence, Ilastik Masks and Wells don't all have the same number of frames")
    #By default, background pixels are labeled as 1, bacteria pixels are 0.
    #Change it so that bacteria are True (1) and background is False (0)
    Wells = Wells < 0.5
    #Remove false object touching the edges of the image
    Imasks = prep_imasks(Imasks,2,Wells)
    return(fluoimage,Imasks,Wells)

def prep_imasks(Imasks,edgedist,Wells):
    '''a- crops regions that don't overlap with the manually labelled wells
    b- crops regions that touch the edge of the plot '''

    for i in range(Imasks.shape[0]):
        Imask = Imasks[i]
        bac_in_wells_ilastik = np.unique(Imask[Wells[i]!=0])
        for bac in np.unique(Imasks):
            if bac not in bac_in_wells_ilastik:
                bacarea = Imask== bac
                Imask[bacarea]=0
    
    numimages,maxy,maxx=Imasks.shape
    for i in range(numimages):
        uedge = np.unique(Imasks[i,edgedist,:])
        ledge = np.unique(Imasks[i,:,edgedist])
        redge = np.unique(Imasks[i,maxy-edgedist,:])
        dedge = np.unique(Imasks[i,:,maxx-edgedist])
        alledge = np.concatenate((uedge, ledge,redge,dedge), axis=None)
        edgebaclist = np.unique(alledge)  
        
        for objnum in edgebaclist:
            for j in range(maxy):
                for k in range(maxx):
                    if Imasks[i,j,k] == objnum:
                        Imasks[i,j,k] = 0
                        
    return(Imasks)
    
    
    
def process_frame(fluoimage, Imask, index, Wells, Dictionary):
    """
    This should contain the main single-frame processing.
    Image and mask should both now be single frames of their parent tiff stacks
    """
    # Show the contour of the first frame's mask over the first image
    [Wells_labels, props_Wells] = wells_props(Imask, Wells, fluoimage)

    Dictionary = data_dict(Imask,Wells, fluoimage, Wells_labels, props_Wells, index, Dictionary)

    return(Dictionary)

def wells_props(Imask,Wells,fluoimage):
    '''
    Extract labeling and region properties information from masks
    '''
    # Convert the binary (True/False) mask to a labelled array where each
    # connected group of nonzero pixels gets assigned a unique number
    Wells_labels= measure.label(Wells)
    #Prepare regionprops for Ilastik and Manual masks
    props_Wells = measure.regionprops(Wells_labels, fluoimage)
    
    
    #Order wells labours by mean X position
    Wells_labels_sorted = orderwells(props_Wells, Wells_labels)
    props_Wells = measure.regionprops(Wells_labels_sorted, fluoimage)
    
    return(Wells_labels_sorted, props_Wells)


def orderwells(props_Wells, Wells_labels):
    '''
    Order well's labels by mean X position
    '''
    #Get a list of each well's mean X position
    Wells_centroid_list = [i.centroid[1] for i in props_Wells]
    #Find the order of each well's mean X position
    Wells_order = np.argsort(Wells_centroid_list)
    #Relabel wells to sort them by mean X position, from left to right
    Wells_labels_sorted = np.zeros(Wells_labels.shape, dtype=int)
    for i in range(len(Wells_centroid_list)):
        Wells_labels_sorted[Wells_labels == Wells_order[i]+1] = i+1 

    return(Wells_labels_sorted)

def data_dict(Imask, Wells, fluoimage, Wells_labels, props_Wells, index, Dictionary):
    '''
    Extracts specific regionprops data for each object and returns it as a dictionary
    '''
#Set up an empty dictionary that will contain all of the data for this frame
    Frame_dict={}
    
    Empty_well = -1
    for well in range(0,int(np.amax(Wells_labels))):

        bac_in_well_ilastik = np.unique(Imask[Wells_labels==well+1])[1:]
        
        if len(bac_in_well_ilastik) == 0:
            Empty_well = well
            Empty_well_x = props_Wells[well].centroid[1]
            Half_well_dx = abs(props_Wells[0].centroid[1]- props_Wells[1].centroid[1])/2
            break
        
    if Empty_well == -1:
        print("Error: no empty well")
    
    
    #For each well
    for well in range(1,int(np.amax(Wells_labels))):
        #Set up four empty dictionaries that will contain all of the data for each object in the well
        Ilastik = {}

        ''' Non-translated '''
        #Find the number of bacteria in the well
        bac_in_well_ilastik = np.unique(Imask[Wells_labels==well])[1:]
        
        #For each Ilastik-identified bacterium in the well
        for bac in bac_in_well_ilastik:

            #Read the bacterium's properties
            standard_bac_mask_Ilastik = Imask==bac
            Imasc_bac_prop = measure.regionprops(standard_bac_mask_Ilastik.astype(int), fluoimage)[0]
            
            #Find the mean intensity of each object
            Ilastik_bac_intensity = Imasc_bac_prop.mean_intensity
            Ilastic_bac_bbox = Imasc_bac_prop.bbox
            Ilastic_bac_length = Ilastic_bac_bbox[3] - Ilastic_bac_bbox[1]
            Ilastik_bac_height = Ilastic_bac_bbox[2] - Ilastic_bac_bbox[0]
            
            Ilastik["Ilastik_bac_{}_intensity".format(bac)] = Ilastik_bac_intensity
            Ilastik["Ilastik_{}_height".format(bac)] = Ilastik_bac_height
            Ilastik["Ilastik_{}_length".format(bac)] = Ilastic_bac_length
            
            if Empty_well != -1:
                
                empty_dx = Empty_well_x - Imasc_bac_prop.centroid[1]
            
                empty_bac_mask_Ilastik = translate_matrix(standard_bac_mask_Ilastik, empty_dx, 0)
                pdms_bac_mask_Ilastik = translate_matrix(standard_bac_mask_Ilastik, empty_dx + Half_well_dx ,0)

                empty_bac_props_Imask = measure.regionprops(empty_bac_mask_Ilastik, fluoimage)
                pdms_bac_props_Imask = measure.regionprops(pdms_bac_mask_Ilastik, fluoimage)
                
                
                Ilastik_empty_intensity = empty_bac_props_Imask[0].mean_intensity
                Ilastik_pdms_intensity = pdms_bac_props_Imask[0].mean_intensity

                Ilastik["Ilastik_empty_{}_intensity".format(bac)]= Ilastik_empty_intensity
                Ilastik["Ilastik_pdms_{}_intensity".format(bac)]= Ilastik_pdms_intensity
                subtracted_Ilastik = Ilastik_bac_intensity - Ilastik_empty_intensity
                if subtracted_Ilastik > 0:
                    Ilastik["Ilastik_subtracted_{}_intensity".format(bac)]= subtracted_Ilastik
                else:
                    Ilastik["Ilastik_subtracted_{}_intensity".format(bac)]= 0
                fluid_Ilastik = Ilastik_empty_intensity - Ilastik_pdms_intensity
                Ilastik["Ilastik_fluid_{}_intensity".format(bac)]= fluid_Ilastik 
    
        #Populate the dictionary
        Frame_dict["Well_{}".format(well)]= {"Ilastik":Ilastik}
    #Add this frame's dictionary to the running dictionary
    Dictionary["Frame_{}".format(index)]= Frame_dict
    return(Dictionary)

def translate_matrix(Matrix, dx,dy):
    '''
    Parameters
    ----------
    Matrix : Matrix
        Matrix to be translated.
    dX : Int
        Number of spaces to shift the matrix horizontally (positive = right).
    dY : Int
        Number of spaces to shift the matrix horizontally (positive = down).

    Returns the matrix translated by dX places horizontally and dY places vertically.
    Note that the function doesn't extend the matrix, and will leave zeros in new empty spaces
    -------

    '''
    dx = int(round(dx))
    dy = int(round(dy))
    shape = Matrix.shape
    NewMatrix =np.zeros(shape, dtype=int )
    for i in range(shape[0]):
        for j in range(shape[1]):
            if i-dy >= 0 and i-dy < shape[0] and j-dx >=0 and j-dx < shape[1]:
                NewMatrix[i,j] = Matrix[i-dy,j-dx]

    return(NewMatrix)


def get_folder_gui():
    ''' This function is used to create the popup window used to determine file directories
    '''
    app = QtWidgets.QApplication([])
    folder = QtWidgets.QFileDialog.getExistingDirectory(
            parent = None, caption = "Select folder")
    return folder

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def extract_data(Data_dictionary,imask_filename,wells_filename):
    '''
    bac_intensity_tracking[:,,]= per bacterium
    bac_intensity_tracking[,:,]= per stat [bac;empty;pdms;resultant;fluid;bbox_height;bbox_length]
    bac_intensity_tracking[,,:]= per frame
    '''
    imasks_bac,imasks_frames,wellsnum = files_read(imask_filename,wells_filename)
    bac_intensity_tracking = np.zeros([max(imasks_bac)+1,7,imasks_frames])
    ioulist = np.zeros([max(imasks_bac)+1,imasks_frames])
    well_bac_num_list = np.zeros([wellsnum,2,imasks_frames])
    framenum = -1
    #Over each frame
    for frame in Data_dictionary.values():
        framenum += 1
        #Over each well
        wellnum = -1
        for well in frame.values():
            wellnum+=1
            Ilastiknum = 0
            Manualnum = 0
            #Over each type of mask [BBox/Ilastik/Manual]
            for mask in well.values():
                #Over every value [intensity/IoU]
                for value in mask:
                    #Extract iou data
                    data_name = value.split("_")
                    bacint= [int(chunk) for chunk in data_name if chunk.isdigit()][0]
                    
                    if "iou" in data_name:
                        if isinstance(mask[value],str):
                            mask[value] = 0
                        ioulist[bacint,framenum] = mask[value]
                    
                    elif "Ilastik" in data_name:
                        Ilastiknum +=1
                        if "bac" in data_name:
                            bac_intensity_tracking[bacint,0,framenum] = mask[value]
                        elif "empty" in data_name:
                            bac_intensity_tracking[bacint,1,framenum] = mask[value]
                        elif "pdms" in data_name:
                            bac_intensity_tracking[bacint,2,framenum] = mask[value]                       
                        elif "subtracted" in data_name:
                            bac_intensity_tracking[bacint,3,framenum] = mask[value]
                        elif "fluid" in data_name:
                            bac_intensity_tracking[bacint,4,framenum] = mask[value] 
                        elif "height" in data_name:
                            bac_intensity_tracking[bacint,5,framenum] = mask[value]
                        elif "length" in data_name:
                            bac_intensity_tracking[bacint,6,framenum] = mask[value]
                        else:
                            raise NameError("oops something went wrong {}".format(data_name))
                    
                    else:
                        raise NameError("oops something went wrong {}".format(data_name))
            well_bac_num_list[wellnum,0,framenum] = Ilastiknum/7
             
    return(ioulist,bac_intensity_tracking,well_bac_num_list)


def Mega_frame_mean_plot(path, Mega_brightnesslist, statname, xname, yname, statnum): 
    
    #Number of frames
    framerange = list(range(len(Mega_brightnesslist[0,0,:])))

    ylist_Ilastik = np.zeros(len(framerange))
    
    #For each frame, extract mean brightness
    for i in framerange:
        ylist_Ilastik[i] = statistics.mean(Mega_brightnesslist[:,statnum,i])
    
    timeslist = np.linspace(0,190,len(framerange))
    #Plot
    plt.figure(figsize = (20,12))
    data_x = timeslist
    data_list_y = ylist_Ilastik
    framelinegraphs(path, data_x,data_list_y, xname, yname, statname)

def mega_single_bacterium_plot(path, brightnesslist):
    
    #Number of frames
    framerange = list(range(len(brightnesslist[0,0,:])))
    
    #Create emty arrays
    bacnum = len(brightnesslist[:,0,0])
    ylist_Ilastik = np.zeros([bacnum,len(framerange)])
    Error_filter = np.ones(bacnum, dtype=bool)
    numtosseddata = 0
    numtossedbac = 0
    for j in range(bacnum):
        zerostracker = 0
        for k in framerange:
            thisframe = brightnesslist[j,3,k]
            if thisframe == 0:
                zerostracker +=1
            ylist_Ilastik[j,k] = thisframe
        if zerostracker >= 0.15*len(framerange):
            Error_filter[j] = False
            numtosseddata += len(framerange) - zerostracker
            numtossedbac += 1
    print(f"Errors: {numtossedbac} bacteria were tossed out of a total of {bacnum}, meaning that {bacnum-numtossedbac} bacteria are remaining. A total of {numtosseddata} nonzero datapoints were tossed, corresponding to {numtosseddata/len(framerange)} per frame. This is an error rate of {numtosseddata/len(framerange)/(bacnum-numtossedbac)}")
    timeslist = np.linspace(0,190,len(framerange))
    
    xname = "Time /min"
    yname = "Fluorescence /a.u."
    stat_name = "all_bac_resultant"
    all_bac_graphs(path, timeslist,ylist_Ilastik,xname,yname,stat_name, brightnesslist, Error_filter, Removelines = True)
    all_bac_graphs(path, timeslist,ylist_Ilastik,xname,yname,"all_bac_errors", brightnesslist, Error_filter)
    

def all_bac_graphs(path, data_x,data_list_y,xname,yname,stat_name, brightnesslist, Error_filter, Removelines = False):     
        fig, ax = plt.subplots(figsize = (20,12))
        for j in range(len(data_list_y[:,0])):
            if Error_filter[j] == True:
                plt.plot(data_x,data_list_y[j,:], color="grey")    
            else: 
                if Removelines== False:
                    plt.plot(data_x,data_list_y[j,:], color="r") 
        #Number of frames
        framerange = list(range(len(brightnesslist[0,0,:])))
        ylist_Ilastik = np.zeros(len(framerange))
        #For each frame, extract mean brightness
        for i in framerange:
            ylist_Ilastik[i] = statistics.mean(brightnesslist[Error_filter,3,i])
        plt.plot(data_x, ylist_Ilastik, "b-s", linewidth=4)
        ax.set_xlabel(xname)
        ax.set_ylabel(yname)
        plt.savefig(os.path.join(path, f"DataOut/{i+1}_{stat_name}_all"))
        plt.close()
        

def framelinegraphs(path, data_x,data_list_y,xname,yname,stat_name):
    fig, ax = plt.subplots(figsize = (20,12))
    plt.plot(data_x,data_list_y)
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.grid(color='grey', linestyle='-', linewidth=1, alpha=0.25)
    plt.savefig(os.path.join(path, f"DataOut/{stat_name}_all"))
    plt.close()


def files_read(imask_filename,wells_filename):
    
    Wells = skio.imread(wells_filename)
    Wells = Wells < 0.5
    Wells_labels= measure.label(Wells[0])
    Wellsnum = len(np.unique(Wells_labels))-1
    
    Imasks = skio.imread(imask_filename)
    Imasks = prep_imasks(Imasks,2,Wells)

    return(np.unique(Imasks),Imasks.shape[0],Wellsnum)


"""  stops anything in the loop from being run if anything from this script is imported  """
if __name__ == "__main__":
    #Set the path of the directory that contains all tiff files
    plt.rcParams.update({'font.size': 25})
    path = get_folder_gui()
    
    createFolder(os.path.join(path,"DataOut"))
    FluoImages= glob.glob(os.path.join(path,"*Fluo.ti*"))[0]
    ManualWells=glob.glob(os.path.join(path,"*Channels.ti*"))[0]
    Masks=glob.glob(os.path.join(path,"*Ilastik.ti*"))[0]
   
    Data_dictionary = main(FluoImages,Masks,ManualWells)

    ioulist,bac_intensity_tracking,well_bac_num_list = extract_data(Data_dictionary,Masks,ManualWells)

    xname = "Time /min"
    yname_list = ["Mean bacterial fluorescence /a.u.", "Mean empty channel fluorescence /a.u.", "Mean pdms fluorescence /a.u.", "Mean resultant bacterial fluorescence /a.u.", "Mean fluid fluorescence /a.u.", "Mean bacterial height /pixel", "Mean bacterial width /pixel"]
    stats_name_list = ["bac_fluorescence","empty_fluorescence","pdms_fluorescence","resultant_fluorescence","fluid_fluorescence","bac_height","bac_length"]
    
    for i in range(len(bac_intensity_tracking[0,:,0])):
        Mega_frame_mean_plot(path, bac_intensity_tracking,stats_name_list[i], xname, yname_list[i], i)
    

    mega_single_bacterium_plot(path, bac_intensity_tracking)
    