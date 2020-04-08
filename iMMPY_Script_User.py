from tqdm import tqdm
from skimage import (io as skio,measure)
from PyQt5 import QtWidgets
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import statistics

def get_folder_gui():
    '''
    Creates a popup window which the user interacts with to determine the file location in which 
    the masks of interest are stored

    Returns
    -------
    folder : String
        This variable stores the path of the folder that the user chooses
    '''
    app = QtWidgets.QApplication([])
    folder = QtWidgets.QFileDialog.getExistingDirectory(
            parent = None, caption = "Select folder")
    return folder

def createFolder(Path):
    '''
    Checks to see if there exists a subfolder called "DataOut". If there isn't, create one
    
    Parameters
    ----------
    Path : String
        The path of the main folder in which the code is directed to run
    '''
    try:
        if not os.path.exists(Path):
            os.makedirs(Path)
    except OSError:
        print ('Error: Creating directory. ' +  Path)

def checksingle(image_list,image_list_name):
    '''
    Checks the number of files found with the previous glob.glob command. If 0, raise and error.
    If more than one, print a warning

    Parameters
    ----------
    image_list : List
        The list of files returned by the previous glob.glob command.
    image_list_name : String
        Name of image type. Only used in error/ warning messages.

    Returns
    -------
     image_list[0] : String
        The precise path of the image.

    '''
    if len(image_list) == 0:
        raise(NameError(f"Error: no {image_list_name} images found"))
    else: 
        if len(image_list) > 1:
            print(f"Warning: More than one {image_list_name} image found")
        return image_list[0]

def main(Input_file_paths):
    '''
    Coordinates the image analysis functions: prepares the data, then sends each frame of it to be pocessed

    Parameters
    ----------
    Input_file_paths : list of strings
        This list contains the paths of each image file used in the analysis: 
            the fluorescent images stack, the ilastik-generated mask, and the manually-generated wells mask

    Returns
    -------
    Data : Dictionary
        A multi-nested dictionary containing the data gathered from image analysis.
    '''

    Imagelist = prep_data(Input_file_paths)

    #Sets up an empty dictionary which will contain all data from this dataset.
    Data_Dict = {}
    
    #for each frame.
    for index in tqdm(range(Imagelist[0].shape[0])):
        Data_Dict = process_frame(index, Imagelist[0][index], Imagelist[1][index],Imagelist[2][index], Data_Dict)
        raise NameError("stop")

    return Data_Dict

def prep_data(Input_file_paths):
    '''
    Read the data and send both masks to be prepared

    Parameters
    ----------
    Input_file_paths : list of strings
        This list contains the paths of each image file used in the analysis: 
            the fluorescent images, the ilastik-generated masks, and the manually-generated wells

    Raises
    ------
    NameError
        This error is raised if the three image sets don't have the same number of frames.

    Returns
    -------
    Imagelist : list of numpy.ndarray
        This list contains the each image file used in the analysis: 
            the fluorescent images stack, the ilastik-generated mask, and the manually-generated wells mask

    '''

    # Load data from their tiffs
    fluoimage = skio.imread(Input_file_paths[0])
    Imask_image = skio.imread(Input_file_paths[1])
    Wells_image = skio.imread(Input_file_paths[2])
    
    if Wells_image.shape[0] != Imask_image.shape[0] != fluoimage.shape[0]:
        raise NameError("Error: Brightfield, Fluorescence, Ilastik Masks and Wells don't all have the same number of frames")
        
    #prepare wells mask
    Wells = prep_wells(Wells_image)

    #prepare ilastik mask
    edgedist = 2
    Imask = trim_imasks(Imask_image,edgedist)
    Imask = imask_check_wells(Imask,Wells)

    Imagelist = [fluoimage,Imask,Wells]

    return(Imagelist)

def prep_wells(Wells_image):
    '''
    Turns the manually generated wells mask into a binary image. As different drawring methods can end up 
    creating masks either the  or background labelled as zero, this function tests both scenarios 
    (only on the first frame, as it is expected that the same method will be maintained throughout).
    

    Parameters
    ----------
    Wells_image : numpy.ndarray
        This array is the array of pixels that makes up the manually-generated well mask

    Returns
    -------
    Wells_binary : numpy.ndarray
        This array is the binary version of the original mask image. It has been altered so that the 
        background pixels have a value of 0, whist the well pixels have a value of 1 .

    '''

    Wells_v1 = Wells_image[0] < 0.5
    Wells_v2 = Wells_image[0] > 0.5

    # Convert the binary (True/False) mask to a labelled array where each
    # connected group of nonzero pixels gets assigned a unique number
    Wells_v1_labels= measure.label(Wells_v1)
    Wells_v2_labels= measure.label(Wells_v2)

    #Compare which version has more unique wells
    if len(np.unique(Wells_v1_labels))> len(np.unique(Wells_v2_labels)):
        Wells_binary = Wells_image < 0.5
    else: 
        Wells_binary = Wells_image > 0.5

    return Wells_binary

def imask_check_wells(Imask,Wells):
    '''
    Deletes any bacteria that fall outside of the manually-drawn wells

    Parameters
    ----------
    Imask : numpy.ndarray
        ilastik-generated mask that shows each bacterium's positions on each frame.
    Well : numpy.ndarray
        manually-generated mask that shows each well's position on each frame.

    Returns
    -------
    Imask: numpy.ndarray
        ilastik-generated mask that shows each bacterium's positions on each frame.

    '''
   
    all_bac = np.unique(Imask)
    #for each frame
    for i in range(Imask.shape[0]):
        Imask_frame = Imask[i]
        Well_frame = Wells[i]
        #find each bacterium that intersects with a well in this frame
        bac_in_wells_ilastik = np.unique(Imask_frame[Well_frame!=0])
        #for each bacterium
        for bac in all_bac:
            #if the bacterium does not intersect with a well in this frame
            if bac not in bac_in_wells_ilastik:
                #delete the bacterium (set its pixels to 0: the number of the background layer)
                bacarea = Imask_frame== bac
                Imask[i][bacarea]=0
    return(Imask)

def trim_imasks(Imask,edgedist):
    '''
    Deletes any bacteria that come within {edgedist} pixels of the edge of the image

    Parameters
    ----------
    Imask : numpy.ndarrays
        Ilastik-generated mask.
    edgedist : integer
        The function will delete any bacteria that come within this many pixels of the edge of the image.

    Returns
    -------
    None.

    '''
    #Find the number of frames and x and y dimensions of the Imask mask
    numframes,maxy,maxx=Imask.shape
    #for each frame
    for i in range(numframes):
        #find each bacterium that comes within {edgedist} pixels of the edge of the image
        uedge = np.unique(Imask[i,edgedist,:])
        ledge = np.unique(Imask[i,:,edgedist])
        redge = np.unique(Imask[i,maxy-edgedist,:])
        dedge = np.unique(Imask[i,:,maxx-edgedist])
        alledge = np.concatenate((uedge, ledge,redge,dedge), axis=None)
        #condense into a single list
        edgebaclist = np.unique(alledge)  
        
        #for each bacterium that comes within {edgedist} pixels of the edge of the image
        for bac in edgebaclist:
            #delete the bacterium (set its pixels to 0: the number of the background layer)
            bacarea = Imask[i] == bac
            Imask[i][bacarea]=0
                        
    return(Imask)
    
def process_frame(index, fluoimage, Imask, Wells, Data_Dict):
    '''
    Call the functions involved in processing each individual frame of the images.

    Parameters
    ----------
    index : integer
        The frame number.
    fluoimage : numpy.ndarray
        Single frame of fluorescent image stack.
    Imask : numpy.ndarray
        Single frame of ilatik-generated bacterial mask.
    Wells : numpy.ndarray
        Single frame of manually-generated well mask.
    Data_Dict : Dictionary
        Dictionary containing all of the data collected on previous frames.

    Returns
    -------
    Data_Dict : Dictionary
        Dictionary containing all of the data collected on previous frames and this one.

    '''

    [Wells_labels, props_Wells] = wells_props(Wells, fluoimage)

    Data_Dict = image_data(Imask, Wells, fluoimage, Wells_labels, props_Wells, index, Data_Dict)

    return(Data_Dict)

def wells_props(Wells,fluoimage):
    '''
    Labels and sorts the well masks, then extracts their properties

    Parameters
    ----------
    Wells : numpy.ndarray
        Single frame of manually-generated well mask.
    fluoimage : numpy.ndarray
        Single frame of fluorescent image stack.

    Returns
    -------    
    Wells_labels_sorted : numpy.ndarray
        Mask of wells, whose labels are ordered by x position, from left to right
    props_Wells : list
        Properties of the sorted well masks

    '''
    
    #Give each well a unique label
    Wells_labels= measure.label(Wells)
    
    #calculate properties of wells
    props_Wells = measure.regionprops(Wells_labels, fluoimage)
    
    #Order wells labours by mean X position
    Wells_labels_sorted = orderwells(props_Wells, Wells_labels)
    
    #calculate properties of wells, which are now ordered
    props_Wells = measure.regionprops(Wells_labels_sorted, fluoimage)
    
    return(Wells_labels_sorted, props_Wells)

def orderwells(props_Wells, Wells_labels):
    '''
    Order a series of wells by x position, from left to right

    Parameters
    ----------
    props_Wells : list
        list of the well's properties.
    Wells_labels : numpy.ndarray
        labelled wells mask.

    Returns
    -------
    Wells_labels_sorted: numpy.ndarray
        Mask of wells, whose labels are ordered by x position, from left to right

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

def image_data(Imask,fluoimage, Wells_labels, props_Wells, index, Data_Dict):
    '''
    Now that all of the masks are prepared, this function overlays them to the experimental fluorescence images
    and collects data, saving it in a nested dictionary

    Parameters
    ----------
    Imask : numpy.ndarray
        ilastik-derived bacterial mask.
    fluoimage : numpy.ndarray
        Experimentally-derived fluorescence image.
    Wells_labels : numpy.ndarray
        labelled manually-drawn wells mask.
    props_Wells : list
        List containing properties of each well.
    index : integer
        Frame number of image and wells.
    Data_Dict : dictionary
        Dictionary containing data from previous frames.

    Returns
    -------
    Data_Dict : dictionary
        Dictionary containing data from previous frames and the current frame.

    '''
    #Determine the prescence of an empty well, as well as some of its properties
    Empty_well,Empty_well_x,Half_well_dx = empty_well_props(Wells_labels, props_Wells, Imask)
    
    #Set up an empty dictionary that will contain all of the data for this frame
    Frame_dict={}
    
    #For each well
    for well in range(1,int(np.amax(Wells_labels))):
        
        #Create a dictionary to store this well's bacteria's data
        Frame_dict[f"Well_{well}"] = {}

        #For each Ilastik-identified bacterium in the well
        bac_in_well_ilastik = np.unique(Imask[Wells_labels==well])[1:]
        for bac in bac_in_well_ilastik:

            #Read the bacterium's properties
            standard_bac_mask_Ilastik = Imask==bac
            Imasc_bac_prop = measure.regionprops(standard_bac_mask_Ilastik.astype(int), fluoimage)[0]
            
            #Find save properties of each object: intensity, height and length
            Ilastik_bac_intensity = Imasc_bac_prop.mean_intensity
            Ilastic_bac_bbox = Imasc_bac_prop.bbox
            Ilastic_bac_length = Ilastic_bac_bbox[3] - Ilastic_bac_bbox[1]
            Ilastik_bac_height = Ilastic_bac_bbox[2] - Ilastic_bac_bbox[0]
            
            #Save these to the dictionary
            Frame_dict[f"Well_{well}"]["Ilastik_bac_{}_intensity".format(bac)] = Ilastik_bac_intensity
            Frame_dict[f"Well_{well}"]["Ilastik_{}_height".format(bac)] = Ilastik_bac_height
            Frame_dict[f"Well_{well}"]["Ilastik_{}_length".format(bac)] = Ilastic_bac_length
            
            #If there is an empty well, continue to more complex analyses
            if Empty_well != -1:
                
                #Translate the bacterium horizontally to the centre of the empty channel
                empty_dx = Empty_well_x - Imasc_bac_prop.centroid[1]
                empty_bac_mask_Ilastik = translate_matrix(standard_bac_mask_Ilastik, empty_dx, 0)
                
                #Record the intensity at this region of an empty channel
                empty_bac_props_Imask = measure.regionprops(empty_bac_mask_Ilastik, fluoimage)
                Ilastik_empty_intensity = empty_bac_props_Imask[0].mean_intensity
                
                #Also record the intensity in between two channels, in the pdms of the chip
                pdms_bac_mask_Ilastik = translate_matrix(standard_bac_mask_Ilastik, empty_dx + Half_well_dx ,0)
                pdms_bac_props_Imask = measure.regionprops(pdms_bac_mask_Ilastik, fluoimage)
                Ilastik_pdms_intensity = pdms_bac_props_Imask[0].mean_intensity
                
                #Save these to the dictionary
                Frame_dict[f"Well_{well}"]["Ilastik_empty_{}_intensity".format(bac)]= Ilastik_empty_intensity
                Frame_dict[f"Well_{well}"]["Ilastik_pdms_{}_intensity".format(bac)]= Ilastik_pdms_intensity
                
                #Calculate two resultant intensities: the resultant bacterial intensity 
                #(intensity of the bacterium - the empty channel background [In theory cannot be lower than zero])...
                subtracted_Ilastik = Ilastik_bac_intensity - Ilastik_empty_intensity
                if subtracted_Ilastik > 0:
                    Frame_dict[f"Well_{well}"]["Ilastik_subtracted_{}_intensity".format(bac)]= subtracted_Ilastik
                else:
                    Frame_dict[f"Well_{well}"]["Ilastik_subtracted_{}_intensity".format(bac)]= 0
                #... and the intensity of the fluid compared to the background pdms
                fluid_Ilastik = Ilastik_empty_intensity - Ilastik_pdms_intensity
                Frame_dict[f"Well_{well}"]["Ilastik_fluid_{}_intensity".format(bac)]= fluid_Ilastik 

    #Add this frame's dictionary to the running dictionary
    Data_Dict["Frame_{}".format(index)]= Frame_dict
    return(Data_Dict)

def empty_well_props(Wells_labels, props_Wells, Imask):
    '''
    Determine the prescence of an empty well, as well as some of its properties

    Parameters
    ----------
    Wells_labels : numpy.ndarray
        Labelled manually-drawn well mask 
    props_Wells : list
        Properties of the sorted well masks
    Imask : numpy.ndarray
        ilastik-drawn bacterial mask

    Returns
    -------
    Empty_well : integer
        Number that corresponds to the label of the empty well. -1 means that there were no empty wells
    Empty_well_x : float
        The x position of the empty well's centroid
    Half_well_dx : float
        Half of the number of pixels in between two channels

    '''
    
    #Set up avariable that will track the existance of an empty well
    Empty_well = -1
    Empty_well_x = 0
    Half_well_dx = 0
    #For each well
    for well in range(0,int(np.amax(Wells_labels))):
        #find each bacterium in that well
        bac_in_well_ilastik = np.unique(Imask[Wells_labels==well+1])[1:]
        #If there aren't any bacteria in the well
        if len(bac_in_well_ilastik) == 0:
            #Record the label and x-position of of the well
            Empty_well = well
            Empty_well_x = props_Wells[well].centroid[1]
            #Calculate half of the inter-well distance
            Half_well_dx = abs(props_Wells[0].centroid[1]- props_Wells[1].centroid[1])/2
            break
    
    if Empty_well == -1:
        print("Warning: no empty well detected. Some of analysis impossible")
        
    return(Empty_well,Empty_well_x,Half_well_dx)

def translate_matrix(Matrix, dx,dy):
    '''
    Returns the matrix translated by dX places horizontally and dY places vertically.
    Note that the function doesn't extend the matrix, and will leave zeros in new empty spaces
    
    Parameters
    ----------
    Matrix : Matrix
        Matrix to be translated.
    dX : float
        Number of spaces to shift the matrix horizontally (positive = right).
    dY : float
        Number of spaces to shift the matrix horizontally (positive = down).

    Returns
    -------
    NewMatrix : Matrix
        Matrix, post-translation.

    '''

    dx = int(round(dx))
    dy = int(round(dy))
    #Create a new matrix of the same size as the old one. Initialise it with zeros
    shape = Matrix.shape
    NewMatrix =np.zeros(shape, dtype=int )
    #For each pixel
    for i in range(shape[0]):
        for j in range(shape[1]):
            #Translate each pixel, as long as it begins and ends in-bounds
            if i-dy >= 0 and i-dy < shape[0] and j-dx >=0 and j-dx < shape[1]:
                NewMatrix[i,j] = Matrix[i-dy,j-dx]

    return(NewMatrix)     

def extract_data(Data_dictionary,imask_filename,wells_filename):
    '''
    Extract data from the nested dictionary produced in the previous section

    Parameters
    ----------
    Data_dictionary : dictionary
        Nested dictionary containing all recorded data.
    imask_filename : string
        The path of the ilastik-generated bacterial mask file 
    wells_filename : string
        The path of the manually-generated wells mask file 

    Raises
    ------
    NameError
        Error raised if a piece of data that can't be interpreted is found.

    Returns
    -------
    bac_data_tracking : matrix
        Matrix containing each piece of extracted data
    imasks_frames : integer
        The number of frames in the images and masks used in the analysis

    '''
    # extract data from masks
    imasks_bac,imasks_frames,wellsnum = files_read(imask_filename,wells_filename)
    
    #create empty matrices to contain the extracted data
    bac_data_tracking = np.zeros([max(imasks_bac)+1,7,imasks_frames])
    
    #initialise the matrices with a ridiculous number
    bac_data_tracking [:,:,:] = -1000000
    

    #For each frame
    framenum = -1
    for frame in Data_dictionary.values():
        framenum += 1
        
        #For each well
        wellnum = -1
        for well in frame.values():
            wellnum+=1
            
            #For each piece of data 
            for value in well:
                
                #Determine bacterial number
                data_name = value.split("_")
                bacint= [int(chunk) for chunk in data_name if chunk.isdigit()][0]
                
                #Look for keywords to extract data from dictionary
                if "Ilastik" in data_name:
                    if "bac" in data_name:
                        bac_data_tracking[bacint,0,framenum] = well[value]
                    elif "empty" in data_name:
                        bac_data_tracking[bacint,1,framenum] = well[value]
                    elif "pdms" in data_name:
                        bac_data_tracking[bacint,2,framenum] = well[value]                       
                    elif "subtracted" in data_name:
                        bac_data_tracking[bacint,3,framenum] = well[value]
                    elif "fluid" in data_name:
                        bac_data_tracking[bacint,4,framenum] = well[value] 
                    elif "height" in data_name:
                        bac_data_tracking[bacint,5,framenum] = well[value]
                    elif "length" in data_name:
                        bac_data_tracking[bacint,6,framenum] = well[value]
                    else:
                        raise NameError("oops something went wrong {}".format(data_name))
                
                else:
                    raise NameError("oops something went wrong {}".format(data_name))
             
    print(type(bac_data_tracking))
    return(bac_data_tracking, imasks_frames)

def files_read(imask_filename,wells_filename):
    '''
    Read the data and send both masks to be prepared. Only specific data is saved

    Parameters
    ----------
    imask_filename : string
        The path of the ilastik-generated bacterial mask file 
    wells_filename : string
        The path of the manually-generated wells mask file 

    Returns
    -------
    baclist : list of integers
        This list contains the label number of each bacterium detected by Ilastik
    framenum : integer
        The number of frames in the images and masks used in the analysis
    Wellsnum : integer
        The number of wells present 
    '''
    Imask_image = skio.imread(Input_file_paths[1])
    Wells_image = skio.imread(Input_file_paths[2])
        
    #prepare wells mask
    Wells = prep_wells(Wells_image)
    
    #Fund number of wells
    Wells_labels= measure.label(Wells[0])
    Wellsnum = len(np.unique(Wells_labels))-1

    #prepare ilastik mask
    edgedist = 2
    Imask = trim_imasks(Imask_image,edgedist)
    Imask = imask_check_wells(Imask,Wells)
    
    baclist = np.unique(Imask)
    framenum = Imask.shape[0]
    

    return(baclist, framenum, Wellsnum)
    
def Frame_mean_plot(path, datalist, statnum, framenum, statname, xname, yname, timeslist = False): 
    '''
    Prepares data for plotting. The mean of a data type will be plotted for each frame.

    Parameters
    ----------
    path : string
        Path of folder containing original images and masks.
    datalist : matrix
        Matrix containing each piece of extracted data.
    statnum : integer
        Integer that corresponds to the data type being plotted. Used to idex the {datalist}.
    framenum : integer
        The number of frames in the original masks and images.
    statname : string
        Name of data type to be plotted.
    xname : string
        X-axis label.
    yname : string
        Y-axis label.
    timeslist : list, optional
        List of times at which each image was taken. If left as the default, False, the 
        graph will instead plot per frame.
    '''
    
    #Initialise an empty array to collect data
    framerange = list(range(framenum))
    ylist_Ilastik = np.zeros(len(framerange))
    
    #For each frame, extract the mean of the data type specified by {statnum}
    for i in framerange:
        ylist_Ilastik[i] = statistics.mean(datalist[:,statnum,i])
    
    #Prepare axes
    if timeslist == False:
       data_x =  framerange
    else:
        data_x = timeslist
    data_y = ylist_Ilastik
    
    #Send data to be plotted
    framelinegraphs(path, data_x, data_y, xname, yname, statname)
    
def framelinegraphs(path, data_x, data_y, xname, yname, stat_name):
    '''
    Plots a simple x/y line graph

    Parameters
    ----------
    path : string
        Path of folder containing original images and masks.
    data_x : list
        x-axis data.
    data_y : list
        y-axis data.
    xname : string
        X-axis label.
    yname : string
        Y-axis label.
    statname : string
        Name of data type to be plotted.

    '''
    fig, ax = plt.subplots(figsize = (20,12))
    plt.plot(data_x,data_y)
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    plt.savefig(os.path.join(path, f"DataOut/{stat_name}_all"))
    plt.close()

def Single_bacterium_brightness_plot(path, datalist, framenum, xname, yname, error_tolerance_percentage = 0, timeslist = False):
    '''
    Prepares data for plotting. Each bacterium's brightness at each frame will be plotted
    
    Parameters
    ----------
    path : string
        Path of folder containing original images and masks.
    datalist : matrix
        Matrix containing each piece of extracted data.
    framenum : integer
        The number of frames in the original masks and images.
    xname : string
        X-axis label.
    yname : string
        Y-axis label.
    error_tolerance_percentage : float, optional
        The highest percentage of erroneous datapoints that are tolerated. The default is 0.
    timeslist : list, optional
        List of times at which each image was taken. If left as the default, False, the 
        graph will instead plot per frame.
    '''
    #Initialise an empty matrix to collect data
    framerange = list(range(framenum))
    bacnum = len(datalist[:,0,0])
    ylist_Ilastik = np.zeros([bacnum,len(framerange)])
    
    #prepare for error detection
    Error_filter = np.ones(bacnum, dtype=bool)
    num_lost_real_data = 0
    num_problematic_bac = 0
    num_errors = 0
    
    #For each bacterium
    for j in range(bacnum):
        #Prepare
        errors_tracker = 0
        #For each frame
        for k in framerange:
            thisframe = datalist[j,3,k]
            #If there was no brightness data collected (value is stil at the default -1000000)
            if thisframe == -1000000:
                #Record this as an error and set its value to zero
                errors_tracker +=1
                thisframe = 0
            #Record value in new matrix
            ylist_Ilastik[j,k] = thisframe
        #If the bacterium has had too many errors (more than error_tolerance %)
        if errors_tracker > 0.01*error_tolerance_percentage*len(framerange):
            #Mark the bacterium as problematic...
            Error_filter[j] = False
            #...and record statistics on number of errors, number of problematic bacteria, 
            #and number of lost non-erroneous pieces of data that belong to those bacteria.
            num_errors += errors_tracker
            num_problematic_bac += 1
            num_lost_real_data += len(framerange) - errors_tracker

    print(f"Errors: {num_problematic_bac} bacteria were tossed out of a total of {bacnum}, meaning that {bacnum-num_problematic_bac} bacteria are remaining. A total of {num_lost_real_data} nonzero datapoints were tossed, corresponding to {num_lost_real_data/len(framerange)} per frame. This is an error rate of {num_lost_real_data/(len(framerange)*(bacnum)-num_errors)}")
    
    #Prepare axes
    if timeslist == False:
       data_x =  framerange
    else:
        data_x = timeslist
    data_y = ylist_Ilastik
    
    all_bac_graphs(path, data_x,data_y,xname,yname,"all_bac_brightness", datalist, framerange, Error_filter, Removelines = True)
    all_bac_graphs(path, data_x,data_y,xname,yname,"all_bac_brightness_errors", datalist, framerange, Error_filter)
    
def all_bac_graphs(path, data_x,data_list_y,xname,yname,stat_name, datalist, framerange, Error_filter, Removelines = False): 
    '''
    Plots each bacterium's brightness at each frame, as well as the mean bacterial brightness.

    Parameters
    ----------    
    path : string
        Path of folder containing original images and masks.
    data_x : list
        x-axis data.
    data_y : list
        y-axis data.
    xname : string
        X-axis label.
    yname : string
        Y-axis label.
    statname : string
        Name of data type to be plotted.
    datalist : matrix
        Matrix containing each piece of extracted data.
    framerange : list
        list of each frame in the original masks and images.
    Error_filter : array of booleans
        Array with each element corresponding to a bacterium. False means that that bacterium is considdered 
        promlematically error-prone
    Removelines : TYPE, optional
        When True, any bacteria labelled problematic by {Error_filter} will be removed from the graph and 
        the calculation of the mean. The default is False.

    '''
    

    ''' step 1: plot the individual bacterium lines'''
        
    fig, ax = plt.subplots(figsize = (20,12))
    #For each bacterium
    for j in range(len(data_list_y[:,0])):
        #If the bacterium isn't problematic, plot it
        if Error_filter[j] == True:
            plt.plot(data_x,data_list_y[j,:], color="grey")   
        #If the bacterium is problematic, plot it in red,
        #but only if {Removelines== False}. Else don't plot it at all
        else: 
            if Removelines== False:
                plt.plot(data_x,data_list_y[j,:], color="r") 
                
                
    '''step 2: plot the mean brightness line'''
    
    #For each frame, extract mean brightness
    ylist_mean_b = np.zeros(len(framerange))
    
    #If {Removelines== False}: considder all bacteria for the mean
    if Removelines== False:
        for i in framerange:
            ylist_mean_b[i] = statistics.mean(datalist[:,3,i])
    #If {Removelines== False}: considder only non-problematic bacteria for the mean
    else:
        for i in framerange:
            ylist_mean_b[i] = statistics.mean(datalist[Error_filter,3,i])
            
    plt.plot(data_x, ylist_mean_b, "b-s", linewidth=4)
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    plt.savefig(os.path.join(path, f"DataOut/{i+1}_{stat_name}_all"))
    plt.close()



"""  this if statement stops anything in the loop from being run if functions from this script are imported into other works """
if __name__ == "__main__":
    
    ''' Step 0: Setup'''
    
    #Set the path of the directory that contains the masks
    path = get_folder_gui()
    
    #Create a folder called "DataOut"
    createFolder(os.path.join(path,"DataOut"))
    
    #Find the specific paths of the three input files. 
    FluoImages= glob.glob(os.path.join(path,"*Fluo.ti*"))
    ManualWells=glob.glob(os.path.join(path,"*Channels.ti*"))
    Masks=glob.glob(os.path.join(path,"*Ilastik.ti*"))
    
    #Check that the right number of files has been found
    FluoImage = checksingle(FluoImages,"fluorescent")
    ManualWell= checksingle(ManualWells,"manual wells")
    imask = checksingle(Masks,"ilastik mask")
    
    Input_file_paths = [FluoImage,imask,ManualWell]
        
    ''' Step 1: Analysis'''
    
    Data_dictionary = main(Input_file_paths)
    
    
    ''' Step 2: Data extraction'''
    
    bac_data_tracking, imasks_frames = extract_data(Data_dictionary,Masks,ManualWells)
    
    
    '''Step 3:  Plotting results'''
    
    plt.rcParams.update({'font.size': 25})
    
    xname = "Time /min"
    yname_list = ["Mean bacterial fluorescence /a.u.", "Mean empty channel fluorescence /a.u.", "Mean pdms fluorescence /a.u.", "Mean resultant bacterial fluorescence /a.u.", "Mean fluid fluorescence /a.u.", "Mean bacterial height /pixel", "Mean bacterial width /pixel"]
    stats_name_list = ["bac_fluorescence","empty_fluorescence","pdms_fluorescence","resultant_fluorescence","fluid_fluorescence","bac_height","bac_length"]
    timeslist = np.linspace(0,190,imasks_frames)
    
    for i in range(len(bac_data_tracking[0,:,0])):
        Frame_mean_plot(path, bac_data_tracking, i, imasks_frames, stats_name_list[i], xname, yname_list[i],  timeslist)
    
    yname = "Fluorescence /a.u."
    Single_bacterium_brightness_plot(path, bac_data_tracking,imasks_frames,timeslist)
    