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
    folder = QtWidgets.QFileDialog.getExistingDirectory(parent = None, caption = "Select folder")
    return folder

def create_folder(path):
    '''
    Checks to see if there exists a subfolder with the given name (in this case, "iMMPY_DataOut"). 
    If there isn't, create one.
    
    Parameters
    ----------
    path : String
        The path of the main folder in which the code is directed to run
    '''
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print ('Error: Creating directory. ' +  path)

def checksingle(image_list, image_list_name):
    '''
    Checks the number of files found with the previous glob.glob command. If 0, raise an error.
    If there is more than one, print a warning

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
        raise NameError(f"Error: no {image_list_name} images found")
    else: 
        if len(image_list) > 1:
            print(f"Warning: More than one {image_list_name} image found")
        return image_list[0]

def iMMPY_Analysis(input_file_paths):
    '''
    Coordinates the image analysis functions: prepares the data, then sends each frame of it to be pocessed

    Parameters
    ----------
    Input_file_paths : list of strings
        This list contains the paths of each image file used in the analysis: 
            the fluorescent images stack, the ilastik-generated mask, and the manually-generated channels mask

    Returns
    -------
    data_dict : Dictionary
        A multi-nested dictionary containing the data gathered from image analysis.
    '''

    image_list = prep_data(input_file_paths, 2)

    #Sets up an empty dictionary which will contain all data from this dataset.
    data_dict = {}
    
    #For each frame of the images.
    for index in tqdm(range(image_list[0].shape[0])):
         frame_dict = process_frame(index, image_list[0][index], image_list[1][index], image_list[2][index])
         data_dict["Frame_{}".format(index)]= frame_dict

    return data_dict , image_list

def prep_data(input_file_paths, edgedist):
    '''
    Read the data and send both masks to be prepared

    Parameters
    ----------
    input_file_paths : list of strings
        This list contains the paths of each image file used in the analysis: 
            the fluorescent images, the ilastik-generated masks, and the manually-generated channels

    Raises
    ------
    NameError
        This error is raised if the three image sets don't have the same number of frames.

    Returns
    -------
    imagelist : list of numpy.ndarray
        This list contains the each image file used in the analysis: 
            the fluorescent images stack, the ilastik-generated mask, and the manually-generated channels mask

    '''

    # Load data from their tiffs
    fluo_image = skio.imread(input_file_paths[0])
    imask_image = skio.imread(input_file_paths[1])
    channels_image = skio.imread(input_file_paths[2])
    
    if channels_image.shape[0] != imask_image.shape[0] != fluo_image.shape[0]:
        raise ValueError("Error: Brightfield, Fluorescence, ilastik Masks and channels don't all have the same number of frames")
        
    #Prepare channels mask
    channels = prep_channels(channels_image, fluo_image)

    #Prepare ilastik mask
    imask = trim_imasks(imask_image, edgedist)
    imask = imask_check_channels(imask, channels)

    imagelist = [fluo_image, imask, channels]

    return imagelist

def prep_channels(channels_image, fluoimage):
    '''
    Turns the manually generated channels mask into a binary image. As different drawring methods can end up 
    creating masks either the  or background labelled as zero, this function tests both scenarios 
    (only on the first frame, as it is expected that the same method will be maintained throughout).

    Parameters
    ----------
    channels_image : numpy.ndarray
        This array is the array of pixels that makes up the manually-generated channel mask
    fluoimage : numpy.ndarray
        Fluorescence image stack.

    Returns
    -------
    channels_binary : numpy.ndarray
        This array is the binary version of the original mask image. It has been altered so that the 
        background pixels have a value of 0, whist the channel pixels have a value of 1 .

    '''
    
    # create two different binary versions of the image, one of which is the negative of the other
    channels_v1 = channels_image < 0.5
    channels_v2 = channels_image > 0.5

    # Convert the binary (True/False) mask to a labelled array where each connected group of 
    #    nonzero pixels gets assigned a unique number
    channels_v1_labels, v1_label_nums = measure.label(channels_v1, return_num = True)
    channels_v2_labels, v2_label_nums = measure.label(channels_v2, return_num = True)

    #Compare which version has more unique channels and set that one as the master version
    if v1_label_nums > v2_label_nums:
        channels_binary = channels_v1_labels
    else: 
        channels_binary = channels_v2_labels
    
    channels_binary = check_channels_duplicate(channels_binary, fluoimage)
    
    return channels_binary

def check_channels_duplicate(channels, fluoimage):
    '''
    Checks channel numbers across each frame to spot inconsistent numbers of channel, 
    and determine if the channels need to be duplicated

    Parameters
    ----------
    channels : numpy.ndarray
        The manually-generated channel mask.
    fluoimage : numpy.ndarray
        The experimentally-derived fluorescent image. Only here to make regionprops work

    Returns
    -------
    channels : numpy.ndarray
        The manually-generated channel mask.
    '''
    
    master_frame_channel_number = -1
    master_frame_frame_number = -1
    duplicate_channels_bool = True
    #For each frame
    for frame in range(channels.shape[0]):
        #Measure the number of channels in the frame
        channels_labels= measure.label(channels[frame])
        channel_num = len(np.unique(channels_labels)) - 1
        #If there's only one: pass
        if channel_num == 1:
            pass
        #If there's none, there's an error
        elif channel_num < 1:
            raise ValueError(f"Warning: frame {frame} has {channel_num} channels marked")
        #If there's more than one:
        else:
            # If this is the first non-one frame, record it's index and its number of channels
            if master_frame_channel_number == -1:
                master_frame_channel_number = channel_num
                master_frame_frame_number = frame
            #If this isn't the first time, check if the number of channels are conserved
            else:
                duplicate_channels_bool = False
                if channel_num == master_frame_channel_number:
                    pass
                else:
                    print(f"Warning: number of channels inconsistent across frames. See frames {master_frame_frame_number} ({master_frame_channel_number}) and {frame}({channel_num})")
    
    # If there was only one frame with more than one channel in it, duplicate it into the other frames
    if duplicate_channels_bool == True:
        channels = duplicate_channels(channels, fluoimage, master_frame_frame_number)

    return(channels)

def duplicate_channels(channels, fluoimage, master_frame_frame_number):
    '''
    Duplicates a single frame's channels onto every other frame, transforming it to account for frameshift

    Parameters
    ----------
    channels : numpy.ndarray
        The manually-generated channel mask.
    fluoimage : numpy.ndarray
        The experimentally-derived fluorescent image. Only here to make regionprops work
    master_frame_frame_number : integer
        The frame index of the master frame.

    Returns
    -------
    channels : numpy.ndarray
        The manually-generated channel mask.

    '''
    print(f"Duplicating channels from frame {master_frame_frame_number}")
    
    #The master frame is the single multi-channeled frame to be copied from.    
    master_frame = channels[master_frame_frame_number]
    
    #Order the channels and measure the leftmost one's centroid position
    mf_labels = measure.label(master_frame)
    mf_properties = measure.regionprops(mf_labels, fluoimage[master_frame_frame_number])
    
    mf_channels_sorted = orderchannels(mf_properties, mf_labels)
    mf_props_sorted = measure.regionprops(mf_channels_sorted, fluoimage[master_frame_frame_number])
    
    mf_channel1_centroid = mf_props_sorted[0].centroid
    
    #For each frame
    for frame in range(channels.shape[0]):
        #If the frame isn't the master frame
        if frame != master_frame_frame_number:
            #Determine the position of the single channel's centroid
            frame_label = measure.label(channels[frame])
            frame_channel_props = measure.regionprops(frame_label, fluoimage[frame])[0]
            frame_channel_centroid = frame_channel_props.centroid
            #Displace the master frame so that its first channel overlapps with this frame's channel
            dy = frame_channel_centroid[0]  - mf_channel1_centroid[0] 
            dx =  frame_channel_centroid[1] - mf_channel1_centroid[1] 
            channels[frame] = translate_matrix(channels[master_frame_frame_number], dx, dy)
    
    return(channels)

def imask_check_channels(imask, channels):
    '''
    Deletes any bacteria that fall outside of the manually-drawn channels

    Parameters
    ----------
    imask : numpy.ndarray
        ilastik-generated mask that shows each bacterium's positions on each frame.
    channel : numpy.ndarray
        manually-generated mask that shows each channel's position on each frame.

    Returns
    -------
    imask: numpy.ndarray
        ilastik-generated mask that shows each bacterium's positions on each frame.

    '''
   
    all_bac = np.unique(imask)
    #For each frame
    for i in range(imask.shape[0]):
        imask_frame = imask[i]
        channel_frame = channels[i]
        #Find each bacterium that intersects with a channel in this frame
        bac_in_channels_ilastik = np.unique(imask_frame[channel_frame!=0])
        #For each bacterium
        for bac in all_bac:
            #If the bacterium does not intersect with a channel in this frame
            if bac not in bac_in_channels_ilastik:
                #Delete the bacterium (set its pixels to 0: the number of the background layer)
                bacarea = imask_frame== bac
                imask[i][bacarea]=0
    return(imask)

def trim_imasks(imask, edgedist):
    '''
    Deletes any bacteria that come within {edgedist} pixels of the edge of the image

    Parameters
    ----------
    imask : numpy.ndarrays
        ilastik-generated mask.
    edgedist : integer
        The function will delete any bacteria that come within this many pixels of the edge of the image.

    Returns
    -------
    imask : numpy.ndarrays
        ilastik-generated mask.

    '''
    #Find the number of frames and x and y dimensions of the imask mask
    numframes,maxy,maxx=imask.shape
    #For each frame
    for i in range(numframes):
        #Find each bacterium that comes within {edgedist} pixels of the edge of the image
        uedge = np.unique(imask[i, edgedist, :])
        ledge = np.unique(imask[i, : ,edgedist])
        redge = np.unique(imask[i, maxy-edgedist, :])
        dedge = np.unique(imask[i, :, maxx-edgedist])
        alledge = np.concatenate((uedge, ledge, redge, dedge), axis=None)
        #Condense into a single list
        edgebaclist = np.unique(alledge)  
        
        #For each bacterium that comes within {edgedist} pixels of the edge of the image
        for bac in edgebaclist:
            for j in range(numframes): 
                #Delete the bacterium (set its pixels to 0: the number of the background layer)
                bacarea = imask[j] == bac
                imask[j][bacarea]=0
                        
    return(imask)
    
def process_frame(index, fluoimage, imask, channels):
    '''
    Call the functions involved in processing each individual frame of the images.

    Parameters
    ----------
    index : integer
        The frame number.
    fluoimage : numpy.ndarray
        Single frame of fluorescent image stack.
    imask : numpy.ndarray
        Single frame of ilatik-generated bacterial mask.
    channels : numpy.ndarray
        Single frame of manually-generated channel mask.

    Returns
    -------
    frame_dict : Dictionary
        Dictionary containing all of the data collected on previous frames and this one.

    '''

    [channels_labels, props_channels] = channels_props(channels, fluoimage)

    frame_dict = image_data(imask, fluoimage, channels_labels, props_channels, index)

    return(frame_dict)

def channels_props(channels,fluoimage):
    '''
    Labels and sorts the channel masks, then extracts their properties

    Parameters
    ----------
    channels : numpy.ndarray
        Single frame of manually-generated channel mask.
    fluoimage : numpy.ndarray
        Single frame of fluorescent image stack.

    Returns
    -------    
    channels_labels_sorted : numpy.ndarray
        Mask of channels, whose labels are ordered by x position, from left to right
    props_channels : list
        Properties of the sorted channel masks

    '''
    
    #Order channels labours by mean X position
    channels_labels_sorted = orderchannels(channels, fluoimage)
    
    #Calculate properties of channels, which are now ordered
    props_channels = measure.regionprops(channels_labels_sorted, fluoimage)
    
    return(channels_labels_sorted, props_channels)

def orderchannels(channels, fluoimage):
    '''
    Labels a series of channels, then orders them by x position, from left to right

    Parameters
    ----------
    channels : numpy.ndarray
        Single frame of manually-generated channel mask.
    fluoimage : numpy.ndarray
        Single frame of fluorescent image stack.

    Returns
    -------
    channels_labels_sorted: numpy.ndarray
        Mask of channels, whose labels are ordered by x position, from left = 1 to right = max (0 is the background)

    '''
    #Give each channel a unique label
    channels_labels= measure.label(channels)
    
    #Calculate properties of channels
    props_channels = measure.regionprops(channels_labels, fluoimage)
    
    #Get a list of each channel's mean X position
    channels_centroid_list = [i.centroid[1] for i in props_channels]
    
    #Find the order of each channel's mean X position
    channels_order = np.argsort(channels_centroid_list)
    
    #Relabel channels to sort them by mean X position, from left to right
    channels_labels_sorted = np.zeros(channels_labels.shape, dtype=int)
    
    for i in range(len(channels_centroid_list)):
        channels_labels_sorted[channels_labels == channels_order[i]+1] = i+1 

    return(channels_labels_sorted)

def image_data(imask, fluoimage, channels_labels, props_channels, index):
    '''
    Now that all of the masks are prepared, this function overlays them to the experimental fluorescence images
    and collects data, saving it in a nested dictionary

    Parameters
    ----------
    imask : numpy.ndarray
        ilastik-derived bacterial mask.
    fluoimage : numpy.ndarray
        Experimentally-derived fluorescence image.
    channels_labels : numpy.ndarray
        labelled manually-drawn channels mask.
    props_channels : list
        List containing properties of each channel.
    index : integer
        Frame number of image and channels.

    Returns
    -------
    frame_dict : dictionary
        Dictionary containing data from the current frame.

    '''
    #Determine the prescence of an empty channel, as channel as some of its properties
    empty_channel, empty_channel_x, half_channel_dx = empty_channel_props(channels_labels, props_channels, imask)
    
    #Set up an empty dictionary that will contain all of the data for this frame
    frame_dict={}
    
    #For each channel
    for channel in range(1,int(np.amax(channels_labels))):
        
        #Create a dictionary to store this channel's bacteria's data
        frame_dict[f"channel_{channel}"] = {}

        #For each ilastik-identified bacterium in the channel
        bac_in_channel_ilastik = np.unique(imask[channels_labels==channel])[1:]
        for bac in bac_in_channel_ilastik:

            #Read the bacterium's properties
            standard_bac_mask_ilastik = imask==bac
            
            imasc_bac_prop = measure.regionprops(standard_bac_mask_ilastik.astype(int), fluoimage)[0]
            
            #Find save properties of each object: intensity, height and length
            ilastik_bac_intensity = imasc_bac_prop.mean_intensity
            ilastic_bac_bbox = imasc_bac_prop.bbox
            ilastic_bac_length = ilastic_bac_bbox[3] - ilastic_bac_bbox[1]
            ilastik_bac_height = ilastic_bac_bbox[2] - ilastic_bac_bbox[0]
            
            #Save these to the dictionary
            frame_dict[f"channel_{channel}"]["ilastik_bac_{}_intensity".format(bac)] = ilastik_bac_intensity
            frame_dict[f"channel_{channel}"]["ilastik_{}_height".format(bac)] = ilastik_bac_height
            frame_dict[f"channel_{channel}"]["ilastik_{}_length".format(bac)] = ilastic_bac_length
            
            #If there is an empty channel, continue to more complex analyses
            if empty_channel != -1:
                
                #Translate the bacterium horizontally to the centre of the empty channel
                empty_dx = empty_channel_x - imasc_bac_prop.centroid[1]
                empty_bac_mask_ilastik = translate_matrix(standard_bac_mask_ilastik, empty_dx, 0).astype(int)
                
                #Record the intensity at this region of an empty channel
                empty_bac_props_imask = measure.regionprops(empty_bac_mask_ilastik, fluoimage)
                ilastik_empty_intensity = empty_bac_props_imask[0].mean_intensity
                
                #Also record the intensity in between two channels, in the pdms of the chip
                pdms_bac_mask_ilastik = translate_matrix(standard_bac_mask_ilastik, empty_dx + half_channel_dx ,0).astype(int)
                pdms_bac_props_imask = measure.regionprops(pdms_bac_mask_ilastik, fluoimage)
                ilastik_pdms_intensity = pdms_bac_props_imask[0].mean_intensity
                
                #Save these to the dictionary
                frame_dict[f"channel_{channel}"]["ilastik_empty_{}_intensity".format(bac)]= ilastik_empty_intensity
                frame_dict[f"channel_{channel}"]["ilastik_pdms_{}_intensity".format(bac)]= ilastik_pdms_intensity
                
                #Calculate two resultant intensities: the resultant bacterial intensity 
                #   (intensity of the bacterium - the empty channel background [In theory cannot be lower than zero])...
                subtracted_ilastik = ilastik_bac_intensity - ilastik_empty_intensity
                if subtracted_ilastik > 0:
                    frame_dict[f"channel_{channel}"]["ilastik_subtracted_{}_intensity".format(bac)]= subtracted_ilastik
                else:
                    frame_dict[f"channel_{channel}"]["ilastik_subtracted_{}_intensity".format(bac)]= 0
                #   ... and the intensity of the fluid compared to the background pdms
                fluid_ilastik = ilastik_empty_intensity - ilastik_pdms_intensity
                frame_dict[f"channel_{channel}"]["ilastik_fluid_{}_intensity".format(bac)]= fluid_ilastik 
    
    return(frame_dict)

def empty_channel_props(channels_labels, props_channels, imask):
    '''
    Determine the prescence of an empty channel, as channel as some of its properties

    Parameters
    ----------
    channels_labels : numpy.ndarray
        Labelled manually-drawn channel mask 
    props_channels : list
        Properties of the sorted channel masks
    imask : numpy.ndarray
        ilastik-drawn bacterial mask

    Returns
    -------
    empty_channel : integer
        Number that corresponds to the label of the empty channel. -1 means that there were no empty channels
    empty_channel_x : float
        The x position of the empty channel's centroid
    half_channel_dx : float
        Half of the number of pixels in between two channels

    '''
    
    #Set up avariable that will track the existance of an empty channel
    empty_channel = -1
    empty_channel_x = 0
    half_channel_dx = 0
    #For each channel
    for channel in range(1,int(np.amax(channels_labels))):
        #Find each bacterium in that channel
        bac_in_channel_ilastik = np.unique(imask[channels_labels==channel+1])[1:]
        #If there aren't any bacteria in the channel
        if len(bac_in_channel_ilastik) == 0:
            #Record the label and x-position of of the channel
            empty_channel = channel
            empty_channel_x = props_channels[channel].centroid[1]
            #Calculate half of the inter-channel distance
            half_channel_dx = abs(props_channels[0].centroid[1]- props_channels[1].centroid[1])/2
            break
    
    if empty_channel == -1:
        print("Warning: no empty channel detected. Some of analysis impossible")
        
    return(empty_channel, empty_channel_x, half_channel_dx)

def translate_matrix(matrix, dx,dy):
    '''
    Returns the matrix translated by dx places horizontally and dy places vertically.
    Note that the function doesn't extend the matrix, and will leave zeros in new empty spaces
    
    Parameters
    ----------
    matrix : matrix
        Matrix to be translated.
    dX : float
        Number of spaces to shift the matrix horizontally (positive = right).
    dY : float
        Number of spaces to shift the matrix horizontally (positive = down).

    Returns
    -------
    new_matrix : Matrix
        Matrix, post-translation.

    '''
    
    dx = int(round(dx))
    adx = abs(dx)
    dy = int(round(dy))
    ady = abs(dy)
    
    #Create a new matrix much bigger than the old one
    y, x = matrix.shape
    big_matrix = np.zeros(( y + 2 * ady, x + 2 * adx))
    
    #Copy the old matrix into the centre of the large matrix
    big_matrix[ady : ady+y, adx : adx+x] = matrix
    
    #Take a section from the large matrix of the same size as the original, but displaced by dx and dy
    new_matrix = big_matrix[ady-dy : ady-dy+y, adx-dx : adx-dx+x]
    
    return(new_matrix)     

def extract_data(Data_dictionary, image_list):
    '''
    Extract data from the nested dictionary produced in the previous section

    Parameters
    ----------
    Data_dictionary : dictionary
        Nested dictionary containing all recorded data.
    imagelist : list of numpy.ndarray
        This list contains the each image file used in the analysis: 
        the fluorescent images stack, the ilastik-generated mask, and the manually-generated channels mask

    Raises
    ------
    NameError
        Error raised if a piece of data that can't be interpreted is found.

    Returns
    -------
    bac_data_tracking : matrix
        Matrix containing each piece of extracted data
    '''
    
    [fluo_image, imask, channels] = image_list 
    
    #Get the number of unique bacteria and the total number of frames
    bac_list = np.unique(imask)
    frame_num = imask.shape[0]

    #Create empty matrices to contain the extracted data
    bac_data_tracking = np.zeros([max(bac_list)+1,7,frame_num])
    
    #Initialise the matrices with a ridiculous number
    bac_data_tracking [:,:,:] = -0.0001
    

    #For each frame
    frame_iter = -1
    for frame in Data_dictionary.values():
        frame_iter += 1
        
        #For each channel
        channel_iter = -1
        for channel in frame.values():
            channel_iter+=1
            
            #For each piece of data 
            for value in channel:
                
                #Determine bacterial number
                data_name = value.split("_")
                bacint= [int(chunk) for chunk in data_name if chunk.isdigit()][0]

                #Look for keywords to extract data from dictionary
                if "bac" in data_name:
                    bac_data_tracking[bacint, 0, frame_iter] = channel[value]
                elif "empty" in data_name:
                    bac_data_tracking[bacint, 1, frame_iter] = channel[value]
                elif "pdms" in data_name:
                    bac_data_tracking[bacint, 2, frame_iter] = channel[value]                     
                elif "subtracted" in data_name:
                    bac_data_tracking[bacint, 3, frame_iter] = channel[value]
                elif "fluid" in data_name:
                    bac_data_tracking[bacint, 4, frame_iter] = channel[value] 
                elif "height" in data_name:
                    bac_data_tracking[bacint, 5, frame_iter] = channel[value]
                elif "length" in data_name:
                    bac_data_tracking[bacint, 6, frame_iter] = channel[value]
                else:
                    raise NameError("oops something went wrong {}".format(data_name))
             
    return(bac_data_tracking)

def Frame_mean_plot(path, datalist, statnum, framenum, statname, xname, yname, timeslist = None): 
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
    ylist_ilastik = np.zeros(len(framerange))
    
    #For each frame, extract the mean of the data type specified by {statnum}
    for i in framerange:
        ylist_ilastik[i] = statistics.mean(datalist[:,statnum,i])
    
    #Prepare axes
    if timeslist is None:
       data_x =  framerange
    else:
        data_x = timeslist
    data_y = ylist_ilastik
    
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
    plt.savefig(os.path.join(path, f"iMMPY_DataOut/{stat_name}_all"))
    plt.close()

def Single_bacterium_brightness_plot(path, datalist, framenum, xname, yname, error_tolerance_percentage = 0, timeslist = None):
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
    frame_range = list(range(framenum))
    bac_num = len(datalist[:,0,0])
    ylist_ilastik = np.zeros([bac_num,len(frame_range)])
    
    #prepare for error detection
    error_filter = np.ones(bac_num, dtype=bool)
    num_lost_real_data = 0
    num_problematic_bac = 0
    num_errors = 0
    
    #For each bacterium
    for j in range(bac_num):
        #Prepare
        errors_tracker = 0
        #For each frame
        for k in frame_range:
            this_frame = datalist[j,3,k]
            #If there was no brightness data collected (value is stil at the default -0.0001)
            if this_frame < 0:
                #Record this as an error and set its value to zero
                errors_tracker +=1
                this_frame = 0
            #Record value in new matrix
            ylist_ilastik[j,k] = this_frame
        #If the bacterium has had too many errors (more than error_tolerance %)
        if errors_tracker > 0.01 * error_tolerance_percentage * len(frame_range):
            #Mark the bacterium as problematic...
            error_filter[j] = False
            #   ...and record statistics on number of errors, number of problematic bacteria, 
            #   and number of lost non-erroneous pieces of data that belong to those bacteria.
            num_errors += errors_tracker
            num_problematic_bac += 1
            num_lost_real_data += len(frame_range) - errors_tracker

    print(f"Errors: {num_problematic_bac} bacteria were tossed out of a total of {bac_num}, meaning that {bac_num-num_problematic_bac} bacteria are remaining. A total of {num_lost_real_data} nonzero datapoints were tossed, corresponding to {num_lost_real_data/len(frame_range)} per frame. This is an error rate of {num_lost_real_data/(len(frame_range)*(bac_num)-num_errors)}")
    
    #Prepare axes
    if timeslist is None:
       data_x =  frame_range
    else:
        data_x = timeslist
    data_y = ylist_ilastik
    
    all_bac_graphs(path, data_x, data_y, xname, yname,"all_bac_brightness", datalist, frame_range, error_filter, removelines = True)
    all_bac_graphs(path, data_x, data_y, xname, yname,"all_bac_brightness_errors", datalist, frame_range, error_filter)
    
def all_bac_graphs(path, data_x, data_list_y, xname, yname, stat_name, datalist, framerange, error_filter, removelines = False): 
    '''
    Plots each bacterium's brightness at each frame, as channel as the mean bacterial brightness.

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
    error_filter : array of booleans
        Array with each element corresponding to a bacterium. False means that that bacterium is considdered 
        promlematically error-prone
    removelines : TYPE, optional
        When True, any bacteria labelled problematic by {Error_filter} will be removed from the graph and 
        the calculation of the mean. The default is False.

    '''

    ''' step 1: plot the individual bacterium lines'''
        
    fig, ax = plt.subplots(figsize = (20,12))
    #For each bacterium
    for j in range(len(data_list_y[:,0])):
        #If the bacterium isn't problematic, plot it
        if error_filter[j] == True:
            plt.plot(data_x, data_list_y[j,:], color="grey")   
        #If the bacterium is problematic, plot it in red,
        #   but only if {Removelines== False}. Else don't plot it at all
        else: 
            if removelines== False:
                plt.plot(data_x,data_list_y[j,:], color="r") 
                
                
    '''step 2: plot the mean brightness line'''
    
    #For each frame, extract mean brightness
    ylist_mean_b = np.zeros(len(framerange))
    
    #If {Removelines== False}: considder all bacteria for the mean
    if removelines== False:
        for i in framerange:
            ylist_mean_b[i] = np.mean(datalist[:,3,i])
    #If {Removelines== False}: considder only non-problematic bacteria for the mean
    else:
        for i in framerange:
            ylist_mean_b[i] = np.mean(datalist[error_filter,3,i])
            
    plt.plot(data_x, ylist_mean_b, "b-s", linewidth=4)
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    plt.savefig(os.path.join(path, f"iMMPY_DataOut/{i+1}_{stat_name}_all"))
    plt.close()

def setup():
    
    #Set the path of the directory that contains the masks
    path = get_folder_gui()
    
    #Create a folder called "iMMPY_DataOut"
    create_folder(os.path.join(path,"iMMPY_DataOut"))
    
    #Find the specific paths of the three input files. 
    fluo_image = glob.glob(os.path.join(path,"*Fluo.ti*"))
    manual_channel = glob.glob(os.path.join(path,"*Channels.ti*"))
    imask = glob.glob(os.path.join(path,"*ilastik.ti*"))
    
    #Check that the right number of files has been found
    fluo_image = checksingle(fluo_image,"fluorescent")
    manual_channel= checksingle(manual_channel,"manual channels")
    imask = checksingle(imask,"ilastik mask")
    
    input_file_paths = [fluo_image, imask, manual_channel]
    
    return input_file_paths, path
    
def plots(path, imask, bac_data_tracking):

    plt.rcParams.update({'font.size': 25})
    
    frame_num = imask.shape[0]
    
    xname = "Time /min"
    yname_list = ["Mean bacterial fluorescence /a.u.", "Mean empty channel fluorescence /a.u.", "Mean pdms fluorescence /a.u.", "Mean resultant bacterial fluorescence /a.u.", "Mean fluid fluorescence /a.u.", "Mean bacterial height /pixel", "Mean bacterial width /pixel"]
    stats_name_list = ["bac_fluorescence","empty_fluorescence","pdms_fluorescence","resultant_fluorescence","fluid_fluorescence","bac_height","bac_length"]
    timeslist = np.linspace(0, 190, frame_num)
    
    for i in range(len(bac_data_tracking[0,:,0])):
        Frame_mean_plot(path, bac_data_tracking, i, frame_num, stats_name_list[i], xname, yname_list[i],  timeslist)
    
    yname = "Fluorescence /a.u."
    Single_bacterium_brightness_plot(path, bac_data_tracking, frame_num, xname, yname, error_tolerance_percentage = 0, timeslist = timeslist)

"""  this if statement stops anything in the loop from being run if functions from this script are imported into other works """
if __name__ == "__main__":
    
    input_file_paths, path = setup()
    
    data_dictionary, image_list = iMMPY_Analysis(input_file_paths)
    
    bac_data_tracking = extract_data(data_dictionary, image_list)
    
    plots(path, image_list[1], bac_data_tracking)