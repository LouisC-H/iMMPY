from tqdm import tqdm
from skimage import (io as skio,measure)
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import statistics
import sys
from PyQt5.QtCore import  Qt
from PyQt5.QtWidgets import QSizePolicy, QDialog, QCheckBox, QFileDialog, QGridLayout, QLabel, QMainWindow, QApplication, QWidget, QPushButton, QLineEdit
from PyQt5.QtGui import QFont

class data_options(list):
    '''
    This object contains data that is accessed and altered by the code
    '''
    path = []
    checkboxes_title = "   Level of output detail"
    misc_options_title = "Miscelaneous options"
    checkboxes_columns_labels = ["Mean data only", "Individual bacteria"]
    checkbox_rows_tooltips = ["Mean brightness of bacteria", "Mean brightness of an empty channel", "Mean brightness of the inter-channel space", "Bacterial brightness - empty channel brightness", "Empty channel brightness - pdms brightness", "Number of pixels that make up a bacterium's vertical size", "Number of pixels that make up a bacterium's horizontal size", "Number of pixels that make up a bacterium's area", "Number of pixels that make up a bacterium's perimiter", "Eccentricity of bacterium, modelling it as an oval", "Major axis length of a bacterium, modelling it as an oval", "Minor axis length of a bacterium, modelling it as an oval", "Orientation of a bacterium, modelling it as an oval",]
    option_labels_tooltips = ["Maximum tolerance of tracking error. 5% means that we tolerate that the bacterium fails to be tracked in up to 5% of frames", "Time at the first frame, used to label the x axis. Probably 0?", "Time at the last frame, used to label the x axis.", "The units of time used above. EG: min, seconds, days... Affects x-axis labelling "]
    nice_names_list = ["Bacterium brightness", "Empty channel brightness", "Pdms brightness", 
                 "Resultant bacterial fluorescence", "Fluid fluorescence", "Height", "Length",
                 "Area", "Perimeter", "Eccentricity", "Major axis length", 
                 "Minor axis length", "Orientation"]
    names_list = ["Bacterium_brightness","Empty_channel_brightness", "Pdms_brightness", 
                 "Resultant_bacterial_fluorescence", "Fluid_fluorescence", "Height", "Length",
                 "Area", "Perimeter", "Eccentricity", "Major_axis_length", 
                 "Minor_axis_length", "Orientation"]
    names_identifyer = ["Bacterium","Empty", "Pdms", "resultant", "fluid", "height", 
                         "length", "Area", "Perimeter", "Eccentricity", "Major", 
                         "Minor", "Orientation"]
    technical_names_list = ["mean_intensity", "mean_intensity", "mean_intensity", 
                            "", "", "", "", "area", "perimeter", "eccentricity", 
                            "major_axis_length", "minor_axis_length", "orientation"]
    defaults_list = [[1, 0], [1, 0], [1, 0], [1, 1], [1, 0], [1, 0], [1, 0], 
                     [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    options_list = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], 
                    [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    error_tolerance = 0
    time_start = []
    time_stop = []
    time_unit = "min"
    save_data = False

class set_labels(QDialog):
    '''
    This class is used to make specifically formatted labels throughout the script
    '''
    def __init__(self):
        
        super().__init__()
        self.initUI()
        
    def initUI(self):
        ''' 
        Creat a default label
        '''
        self.label = QLabel(self)
        self.label.setText("Placeholder")
        self.label.adjustSize()
        
    def set_label_title(self, name, size):
        '''
        Formats the label as a section title

        Parameters
        ----------
        name : String
            Name of label.
        size : Int
            Horizontal width of label.
        '''
        self.label.setText(name)
        self.label.setFixedWidth(size)
        myFont = QFont()
        myFont.setBold(True)
        self.label.setFont(myFont)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.adjustSize()
    
    def set_label_dynamic(self, name):
        '''
        Formats the label dynamically according to the text size

        Parameters
        ----------
        name : String
            Name of label.
        '''
        self.label.setText(name)
        self.label.adjustSize()
    
    def set_label_right(self, name):
        '''
        Formats the label dynamically as a right-justified checkbox row label

        Parameters
        ----------
        name : String
            Name of label.
        '''
        self.label.setText(name)
        self.label.setFixedWidth(185)
        self.label.setAlignment(Qt.AlignRight)
        
    def set_label_tooltip(self, tool_text):
        '''
        Modify the label with a hover tooltip
        '''
        self.setToolTip(tool_text)

class cb_col_labels(QWidget):
    '''
    This object is a widget that creates the checkbox column labels
    '''
    def __init__(self):
        
        super().__init__()
        self.initUI()
        
    def initUI(self):
        ''' 
        Prepare the grid and two labels to go onto it
        '''
        grid = QGridLayout()
        self.setLayout(grid)
        
        lb0 = set_labels()
        lb0.set_label_dynamic(data_options.checkboxes_columns_labels[0])
        
        lb1 = set_labels()
        lb1.set_label_dynamic(data_options.checkboxes_columns_labels[1])
        
        grid.addWidget(lb0, 0, 0)
        grid.addWidget(lb1, 0, 1)
       
class checkboxes(QWidget):
    ''' 
    Create a row of two checkboxes to be used to alter options
    '''
    def __init__(self, row):
        '''
        Parameters
        ----------
        row : int
            The row at which these checkboxes were created.
        '''
        super().__init__()
        self.row = row
        self.initUI()
        
    def initUI(self):
        ''' 
        Prepare the grid and create two checkboxes
        '''
        grid = QGridLayout()
        self.setLayout(grid)
        
        self.cb0 = QCheckBox(self)
        self.cb0.stateChanged.connect(self.cb0_change)
        
        self.cb1 = QCheckBox(self)
        self.cb1.stateChanged.connect(self.cb1_change)
        
        grid.addWidget(self.cb0, 0, 1)
        grid.addWidget(self.cb1, 0, 3)
    
    def set_checkbox_tooltip(self, tool_text):
        '''
        Modify the checkboxes with a hover tooltip
        '''
        self.cb0.setToolTip(tool_text)
        self.cb1.setToolTip(tool_text)
    
    def default_checks(self, defaults):
        '''
        Toggle the checkboxes so that they match the default options
        
        Parameters
        ----------
        defaults : list of ints
            Binary list, with ones corresponding to a checkbox toggled on.

        '''
        if defaults[0] == 1:
            self.cb0.toggle()
        if defaults[1] == 1:
            self.cb1.toggle()
    
    def cb0_change(self):
        '''
        If checkbox 0 is toggled, change the corresponding option stored in data_options
        '''
        self.change_options(self.row, 0)              
    
    def cb1_change(self):
        '''
        If checkbox 1 is toggled, change the corresponding option stored in data_options.
        Additionally, if it is being toggled on and checkbox 0 is currently unticked, toggle it.
        '''
        self.change_options(self.row, 1)
        
        if self.cb1.isChecked():
            if self.cb0.isChecked():
                return
            else:
                self.cb0.toggle()   
    
    def change_options(self, row, column):
        '''
        Once a checkbox has been ticked, change the corresponding option stored stored in data_options

        Parameters
        ----------
        row : int
            The row of the checkbox.
        column : int
            The column of the checkbox.
        '''
        options_data = data_options.options_list[row][column]
        
        if options_data == 0:
            options_data = 1
        elif options_data == 1: 
            options_data = 0
        else:
            raise ValueError(f"something's gone wrong with the options: option {row} {column} == {options_data}")
        
        data_options.options_list[row][column] = options_data

class checkboxes_gui(QWidget):

    def __init__(self,  title, names_list, defaults_list):
        '''
        Parameters
        ----------
        title : String
            Title of the checkboxes section.
        names_list : List
            List of names of each column.
        defaults_list : List
            List of the default on/off positions of the checkboxes.
        '''
        super().__init__()
        
        self.title = title
        self.names_list = names_list
        self.defaults_list = defaults_list
        
        self.initUI()
        
    def initUI(self):
        '''
        Prepare the grid and add the title, checkboxes, and column/row labels
        '''
        grid = QGridLayout()
        self.setLayout(grid)
        
        # Iterate over each row of the layout grid:
        for i in range(len(self.names_list)+2):
            
            if i == 0:
                #Set the section title
                lab_title = set_labels()
                lab_title.set_label_title(self.title, 500)
                grid.addWidget(lab_title, i, 0, 1, 2)
            
            elif i == 1:
                #Set the column labels
                column_label = cb_col_labels()
                grid.addWidget(column_label, i, 1)
            
            else: 
                #Create the main checkbox grid:
                #Set labels
                label_list = set_labels()
                label_list.set_label_right(self.names_list[i-2])
                label_list.set_label_tooltip(data_options.checkbox_rows_tooltips[i-2])
                grid.addWidget(label_list, i, 0)
                
                #Create checkboxes
                box_list = checkboxes(i-2)
                box_list.default_checks(self.defaults_list[i-2])
                box_list.set_checkbox_tooltip(data_options.checkbox_rows_tooltips[i-2])
                grid.addWidget(box_list, i, 1)

class label_and_qle(QWidget):
    '''
    This class is used to create a label paired with a text input box, and is
    used to allow the user to adjust options
    '''
    
    def __init__(self, label, data_options, defaut, tooltip):
        '''
        Parameters
        ----------
        label : String
            The label's text.
        data_options : String
            String detailing the option that the text box is linked too.
        defaut : String
            The default value of the text box.
        tooltip : String
            The text of the label's hover-over tooltip.
        '''
        super().__init__()
        self.label = label
        self.data_options = data_options
        self.defaut = defaut
        self.tooltip  = tooltip
        self.initUI()
    
    def initUI(self):
        '''
        Prepare the label and text box and arrange them horizontally
        '''
        grid = QGridLayout()
        self.setLayout(grid)

        qle = QLineEdit(self)
        qle.setPlaceholderText(f"default = {self.defaut}")
        qle_label = set_labels()
        qle_label.set_label_dynamic(self.label)
        qle_label.set_label_tooltip(self.tooltip)
        
        #Stop the text box from expanding to take up the whole window
        sizePolicy = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        qle.setSizePolicy(sizePolicy)

        qle.textChanged[str].connect(self.change_options)
        
        grid.addWidget(qle_label, 0, 0, 1, 2)
        grid.addWidget(qle, 0, 2, 1, 1)
        
    def change_options(self, text):
        '''
        As the user types, the relevant data_options option is updated

        Parameters
        ----------
        text : String
            DESCRIPTION.
        '''
        
        #This is the only option for which we expect a string input, so it must be treated differently
        if self.data_options == "data_options.time_unit":
            data_options.time_unit = text
        
        #For all other cases:
        else:
            #If the text block is blank, reset it to its default state
            if text == "":
                exec(f"{self.data_options} = []")
            #
            else:
                if text.isdigit() == False:
                    print(f"Warning: {self.data_options} == {text}. This isn't a number")
                exec(f"{self.data_options} = int({text})")

class cb_single(QWidget):
    ''' 
    Create a row of two checkboxes to be used to alter options
    '''
    def __init__(self, label, tooltip):
        '''
        Parameters
        ----------
        row : int
            The row at which these checkboxes were created.
        '''
        super().__init__()
        self.label = label
        self.tooltip = tooltip
        self.initUI()
        
    def initUI(self):
        ''' 
        Prepare the grid and create two checkboxes
        '''
        grid = QGridLayout()
        self.setLayout(grid)
        
        
        cb_label =  set_labels()
        cb_label.set_label_dynamic(self.label)
        cb_label.set_label_tooltip(self.tooltip)
        
        self.cb = QCheckBox(self)
        self.cb.setToolTip(self.tooltip)
        self.cb.stateChanged.connect(self.change_options)
        
        grid.addWidget(cb_label, 0, 0)
        grid.addWidget(self.cb, 0, 1)          
    
    def change_options(self):
        '''
        Once a checkbox has been ticked, change the corresponding option stored stored in data_options

        Parameters
        ----------
        row : int
            The row of the checkbox.
        column : int
            The column of the checkbox.
        '''
        options_data = data_options.save_data
        
        if options_data == False:
            options_data = True
        elif options_data == True: 
            options_data = False
            
        data_options.save_data = options_data

class misc_options_gui(QWidget):
        
    def __init__(self):
        
        super().__init__()
        self.initUI()
        
    def initUI(self):
        '''
        Generate the grid, then add the title, options text boxes and labels.

        '''
        grid = QGridLayout()
        self.setLayout(grid)
        
        #Set the section title
        lab_title = set_labels()
        lab_title.set_label_title(data_options.misc_options_title, 300)
        grid.addWidget(lab_title, 0, 0, 3, 1)
        #Note, extends downwards to contain the text boxes
        
        ext_data = cb_single("Extract data?", "If ticked, data will additionally be saved as a csv. See protocol for more information.")
        grid.addWidget(ext_data, 3, 0, 3, 1)
        
        error_tol = label_and_qle("Error tolerance ", "data_options.error_tolerance", "0 %", data_options.option_labels_tooltips[0])
        grid.addWidget(error_tol, 6, 0)
                
        spacer_label = set_labels()
        spacer_label.set_label_title("", 300)
        grid.addWidget(spacer_label, 7, 0, 1, 1)

        
        time_start = label_and_qle("Time at start ", "data_options.time_start", "0", data_options.option_labels_tooltips[1])
        grid.addWidget(time_start, 8, 0)
        time_stop = label_and_qle("Time at end ", "data_options.time_stop", "# of frames", data_options.option_labels_tooltips[2])
        grid.addWidget(time_stop, 9, 0)
        time_units = label_and_qle("Units of time ", "data_options.time_unit", "min", data_options.option_labels_tooltips[3])
        grid.addWidget(time_units, 10, 0)
        
        time_label = set_labels()
        time_label.set_label_dynamic("  Note: If no final time is given, graphs will be plotted  \n          against frame number rather than time")
        grid.addWidget(time_label, 11, 0, 2, 1)
        
        #Creat a button to set Folder destination
        set_path =  QPushButton('set folder', self)
        set_path.clicked.connect(Options_Menu.get_folder)
        grid.addWidget(set_path, 12, 0)
        
        #Create button to launch iMMPY program
        run_button = QPushButton('Run iMMPY script', self)
        run_button.clicked.connect(Options_Menu.start_iMMPY)
        grid.addWidget(run_button, 14, 0)
    
class Options_Menu(QMainWindow):
    '''
    This object is the main window of the options menu
    '''
    
    def __init__(self):
        
        super().__init__()
        self.initUI()
     
    def initUI(self):
        '''
        Initialise main window's UI as two side-by-side columns.
        '''
        
        #Create central widget
        self.central_widget = QWidget()               
        self.setCentralWidget(self.central_widget) 
        
        #Define grid layout
        self.grid = QGridLayout()
        self.centralWidget().setLayout(self.grid)
        
        #Add the checkboxe column and the misc options column
        checkboxes = checkboxes_gui(data_options.checkboxes_title, data_options.names_list, data_options.defaults_list)
        misc_options = misc_options_gui()
        self.grid.addWidget(checkboxes, 0, 0, 1, 3)
        self.grid.addWidget(misc_options, 0, 3, 1, 2)
        
        #Set window properties
        self.setGeometry(600, 200, 900, 750)
        self.setWindowTitle('Options menu')
        self.show()
    
    def start_iMMPY(self):
        '''
        When the "Run iMMPY script" button is pushed, start the main iMMPY script
        '''
        #If the path hasn't been set yet, promt the user to do that
        if data_options.path == []:
            Options_Menu.get_folder()
        
        else:
            #Minimise the main window, run the script, then close the window once the script has finished
            QApplication.activeWindow().showMinimized()
            run_iMMPY()
            QApplication.activeWindow().close()
        
    def get_folder():
        '''
        When the "set folder" button is pushed, prompt the user to set a file path
        '''
        folder = QFileDialog.getExistingDirectory(parent = None, caption = "Select folder")
        data_options.path = folder

def main():
    app = QApplication(sys.argv)
    ex = Options_Menu()
    sys.exit(app.exec_())
    
def run_iMMPY():
    '''
    This function runs the four main phases of the script, one after the other

    '''
    print("Running iMMPY script")
    
    input_file_paths, path = setup()
    
    data_dictionary, image_list = iMMPY_Analysis(input_file_paths)
    
    bac_data_tracking = extract_data(data_dictionary, image_list, path)
    
    plots(path, image_list[1], bac_data_tracking)
    
def setup():
    '''
    This function uses the path set in the options menu to find and check the three 
    input files

    Returns
    -------
    input_file_paths : list
        list of the paths of the three files.
    path : string
        path of the directory that contains the files.

    '''
    #Set the path of the directory that contains the masks
    path = data_options.path
    
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
    ValueError
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
    
    #For each frame:
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
    mf_channels_sorted = orderchannels(master_frame, fluoimage[0])
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

    frame_dict = collect_image_data(imask, fluoimage, channels_labels, props_channels, index)

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

def collect_image_data(imask, fluoimage, channels_labels, props_channels, index):
    '''
    Now that all of the masks are prepared, this function calls for the data of interest to be extracted,
    recording them in a nested dictionary

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
    #List of the options set by the user using the GUI
    options_list = data_options.options_list
    
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
            imask_bac_prop = measure.regionprops(standard_bac_mask_ilastik.astype(int), fluoimage)[0]
            ilastik_bac_bbox = imask_bac_prop.bbox
                
            #Call the functions to write the data of interest into the nested dictionary (determined by the options_list)
            for i in ([0, 7, 8, 9, 10, 11, 12]):
                if sum(options_list[i]) > 0:
                    frame_dict = collect_bac_props(frame_dict, imask_bac_prop, i, channel, bac)
                            
            if sum(options_list[5]) > 0:
                collect_bac_length(frame_dict, ilastik_bac_bbox, channel, bac)
                
            if sum(options_list[6]) > 0:
                collect_bac_height(frame_dict, ilastik_bac_bbox, channel, bac)
  
            if empty_channel != -1:
                
                #If there is an empty channel, continue to more complex analyses               
                empty_dx = empty_channel_x - imask_bac_prop.centroid[1]
                frame_dict = collect_complex_brightnesses(frame_dict, standard_bac_mask_ilastik, fluoimage, options_list, imask_bac_prop, empty_dx, half_channel_dx, channel, bac)

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

def collect_bac_props(frame_dict, bacterium_props, property_index, channel, bac):
    '''
    Collects data on one bacterium's properties and adds it to the running dictionary

    Parameters
    ----------
    frame_dict : dictionary
        Dictionary containing data from the current frame.
    bacterium_props : list
        List of the properties of the bacterium.
    property_index : integer 
        Integer used to index the properties from a list.
    channel : integer
        Integer that stores the index of the channel in which the bacterium is found.
    bac : integer
        Integer that stores the index of the bacterium.

    Returns
    -------
    frame_dict : dictionary
        Dictionary containing data from the current frame.
    '''
    prop_name = data_options.names_list[property_index]
    prop_technical_name = data_options.technical_names_list[property_index]
    ilastik_bac_property = []
    ilastik_bac_property = eval(f"bacterium_props.{prop_technical_name}")
    frame_dict[f"channel_{channel}"]["ilastik_bac_{}_".format(bac) + prop_name] = ilastik_bac_property
    return frame_dict
    
def collect_bac_length(frame_dict, ilastik_bac_bbox, channel, bac):
    '''
    Measures one bacterium's length and adds it to the running dictionary.

    Parameters
    ----------
    frame_dict : dictionary
        Dictionary containing data from the current frame.
    ilastik_bac_bbox : list
        List of the bacterium's bounding box's properties.
    channel : integer
        Integer that stores the index of the channel in which the bacterium is found.
    bac : integer
        Integer that stores the index of the bacterium.

    Returns
    -------
    frame_dict : dictionary
        Dictionary containing data from the current frame.
    '''
    ilastik_bac_length = ilastik_bac_bbox[3] - ilastik_bac_bbox[1] 
    frame_dict[f"channel_{channel}"]["ilastik_bac_{}_length".format(bac)] = ilastik_bac_length
    return frame_dict
    
def collect_bac_height(frame_dict, ilastik_bac_bbox, channel, bac):
    '''
    Measures one bacterium's length and adds it to the running dictionary.

    Parameters
    ----------
    frame_dict : dictionary
        Dictionary containing data from the current frame.
    ilastik_bac_bbox : list
        List of the bacterium's bounding box's properties.
    channel : integer
        Integer that stores the index of the channel in which the bacterium is found.
    bac : integer
        Integer that stores the index of the bacterium.

    Returns
    -------
    frame_dict : dictionary
        Dictionary containing data from the current frame.
    '''
    ilastik_bac_height = ilastik_bac_bbox[2] - ilastik_bac_bbox[0]
    frame_dict[f"channel_{channel}"]["ilastik_bac_{}_height".format(bac)] = ilastik_bac_height
    return frame_dict

def collect_complex_brightnesses(frame_dict, standard_bac_mask_ilastik, fluoimage, options_list, imask_bac_prop, empty_dx, half_channel_dx, channel, bac):
    '''
    This function deals with calculating the more complex bacterial properties
    
    Parameters
    ----------
    frame_dict : dictionary
        Dictionary containing data from the current frame.
    standard_bac_mask_ilastik : numpy.ndarray
        ilastik-dreived mask labelling the pixels that make up the bacterium
    fluoimage : numpy.ndarray
        Experimentally-derived fluorescence image.
    options_list : list
        List of the options set by the user using the GUI.
    imask_bac_prop : list
        list of properties of the bacterium.
    empty_dx : float
        Number of pixels in the x axis between the bacterium and the empty channel.
    half_channel_dx : float
        Half of the number of pixels in between two channels
    channel : integer
        Integer that stores the index of the channel in which the bacterium is found.
    bac : integer
        Integer that stores the index of the bacterium.

    Returns
    -------
    frame_dict : dictionary
        Dictionary containing data from the current frame.

    '''
    if sum(options_list[1]) > 0 or sum(options_list[3]) > 0 or sum(options_list[4]) > 0:
        
        #Translate the bacterium horizontally to the centre of the empty channel
        empty_bac_mask_ilastik = translate_matrix(standard_bac_mask_ilastik, empty_dx, 0).astype(int)
        
        #Record the intensity at this region of an empty channel
        empty_bac_props_imask = measure.regionprops(empty_bac_mask_ilastik, fluoimage)[0]
        ilastik_empty_intensity = empty_bac_props_imask.mean_intensity
        
        if sum(options_list[1]) > 0:
            frame_dict = collect_bac_props(frame_dict, empty_bac_props_imask, 1, channel, bac)
        if sum(options_list[3]) > 0:
            ilastik_bac_intensity = imask_bac_prop.mean_intensity
            frame_dict = collect_resultant_brightness(frame_dict, ilastik_bac_intensity, ilastik_empty_intensity, channel, bac)
        
        
    if sum(options_list[4]) > 0:
        #Also record the intensity in between two channels, in the pdms of the chip
        pdms_bac_mask_ilastik = translate_matrix(standard_bac_mask_ilastik, empty_dx + half_channel_dx ,0).astype(int)
        
        pdms_bac_props_imask = measure.regionprops(pdms_bac_mask_ilastik, fluoimage)[0]
        ilastik_pdms_intensity = pdms_bac_props_imask.mean_intensity
        
        if sum(options_list[2]) > 0:
            frame_dict = collect_bac_props(frame_dict, pdms_bac_props_imask, 2, channel, bac)
        if sum(options_list[4]) > 0:
            frame_dict = collect_fluid_brightness(frame_dict, ilastik_pdms_intensity, ilastik_empty_intensity, channel, bac)                

    return frame_dict

def collect_resultant_brightness(frame_dict, ilastik_bac_intensity, ilastik_empty_intensity, channel, bac):
    '''
    Determines the difference between the intensity of a bacterium and the intensity of an equivalent region
    in an empty channel. (Results of less than zero are thought to be the result of noise as it is expected that 
    bacteria should always be at least as bright as an empty channel. These results are therefore set to zero.)

    Parameters
    ----------
    frame_dict : dictionary
        Dictionary containing data from the current frame.
    ilastik_bac_intensity : float
        Mean intensity of pixels within the bounds of the bacterium.
    ilastik_empty_intensity : float
        Mean intensity of pixels in an equivalent region of an empty channel.
    channel : integer
        Integer that stores the index of the channel in which the bacterium is found.
    bac : integer
        Integer that stores the index of the bacterium.

    Returns
    -------
    frame_dict : dictionary
        Dictionary containing data from the current frame.
    '''
    resultant_ilastik = ilastik_bac_intensity - ilastik_empty_intensity
    if resultant_ilastik > 0:
        frame_dict[f"channel_{channel}"]["ilastik_resultant_{}_intensity".format(bac)]= resultant_ilastik
    else:
        frame_dict[f"channel_{channel}"]["ilastik_resultant_{}_intensity".format(bac)]= 0
    return frame_dict

def collect_fluid_brightness(frame_dict, ilastik_pdms_intensity, ilastik_empty_intensity, channel, bac):
    '''
    Determines the difference between the intensity of an empty channel and the intensity of an equivalent region
    of the space in-between channels. This is equivalent to the brightness of the fluid relative to the pdms background

    Parameters
    ----------
    frame_dict : dictionary
        Dictionary containing data from the current frame.
    ilastik_bac_intensity : float
        Mean intensity of pixels within the bounds of the bacterium.
    ilastik_empty_intensity : float
        Mean intensity of pixels in an equivalent region of an empty channel.
    channel : integer
        Integer that stores the index of the channel in which the bacterium is found.
    bac : integer
        Integer that stores the index of the bacterium.

    Returns
    -------
    frame_dict : dictionary
        Dictionary containing data from the current frame.
    '''
    fluid_ilastik = ilastik_empty_intensity - ilastik_pdms_intensity
    frame_dict[f"channel_{channel}"]["ilastik_fluid_{}_intensity".format(bac)] = fluid_ilastik 
    return frame_dict

def extract_data(Data_dictionary, image_list, path):
    '''
    Extract data from the nested dictionary produced in the previous section

    Parameters
    ----------
    Data_dictionary : dictionary
        Nested dictionary containing all recorded data.
    imagelist : list of numpy.ndarray
        This list contains the each image file used in the analysis: 
        the fluorescent images stack, the ilastik-generated mask, and the manually-generated channels mask
    path : String
        Path of directory containing the original input files.

    Returns
    -------
    bac_data_tracking : array
        Matrix containing each piece of extracted data
    '''

    [fluo_image, imask, channels] = image_list 
    
    identifyer_list = data_options.names_identifyer
    
    #Get the number of unique bacteria and the total number of frames
    bac_list = np.unique(imask)
    frame_num = imask.shape[0]

    #Create empty matrices to contain the extracted data
    bac_data_tracking = np.zeros([max(bac_list)+1, len(identifyer_list), frame_num])
    
    #Initialise the matrices with an impossible number
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
                for i in range(len(identifyer_list)):
                    if identifyer_list[i] in data_name:
                        
                        bac_data_tracking[bacint, i, frame_iter] = channel[value]
    
    if data_options.save_data == True:
        export_data(bac_data_tracking, path)
    
    return(bac_data_tracking)

def export_data(data, path):
    '''
    Changes the data from a 3D to a 2D array, then exports it as a csv file

    Parameters
    ----------
    bac_data_tracking : array
        Matrix containing each piece of extracted data
    path : String
        Path of directory containing the original input files.
    '''

    bacnum = len(data[:,1,1])
    statnum = len(data[1,:,1])
    framenum =len(data[1,1,:])
    data_2D = np.zeros([bacnum,statnum*framenum])
    for frame in range(framenum):
        for stat in range(statnum):
            for bac in range(bacnum):
                #Turn [bac,stats,frames] array into [bac, [statsframe1, statsframe2, statsframe3, ...]]
                data_2D[bac, stat+statnum*frame] = data[bac, stat, frame]
            
    print("Exporting data")
    
    np.savetxt(os.path.join(path, f"iMMPY_DataOut/exported_data.csv"), data_2D, delimiter=",")

def plots(path, imask, bac_data_tracking):
    '''
    This function uses the extracted data to create graphs according to the user's choices in the options menu

    Parameters
    ----------
    path : String
        Path of directory containing the original input files.
    imask : matrix/image
        Image of the ilastik-generated mask.
    bac_data_tracking : Matrix
        Matrix containing all of the extracted data.
    '''
    plt.rcParams.update({'font.size': 25})
    
    frame_num = imask.shape[0]
    
    # If the user has given a final time, plot time on the x axis
    if data_options.time_stop != []:
        times_list = np.linspace(data_options.time_start, data_options.time_stop, frame_num)
        time_unit = data_options.time_unit
        x_name = "Time /" + time_unit
    #If no time has been set by the user, plot frame number on the x axis
    else: 
        times_list = list(range(frame_num))
        time_unit = "frame"
        x_name = "Frame"
    
    #If the user has not given an error tolerance, assume it to be zero
    if data_options.error_tolerance == []:
        data_options.error_tolerance = 0
    
    options_list = data_options.options_list
    names_list = data_options.names_list
    
    #First plot mean data
    for i in range(len(options_list)):
        if options_list[i][0] == 1:
            Frame_mean_plot(path, bac_data_tracking, i, frame_num, names_list[i], x_name, names_list[i],  times_list)
    #Then plot individual bacteria data
    for i in range(len(options_list)):
        if options_list[i][1] == 1:
            Single_bacterium_brightness_plot(path, bac_data_tracking, i, frame_num, names_list[i], x_name, names_list[i], times_list, data_options.error_tolerance)  

def Frame_mean_plot(path, data_list, stat_num, frame_num, stat_name, x_name, y_name, times_list): 
    '''
    Prepares data for plotting. The mean of a data type will be plotted for each frame.

    Parameters
    ----------
    path : string
        Path of folder containing original images and masks.
    data_list : matrix
        Matrix containing each piece of extracted data.
    stat_num : integer
        Integer that corresponds to the data type being plotted. Used to idex the {datalist}.
    frame_num : integer
        The number of frames in the original masks and images.
    stat_name : string
        Name of data type to be plotted.
    x_name : string
        X-axis label.
    y_name : string
        Y-axis label.
    times_list : list, optional
        List of times at which each image was taken. If left as the default, False, the 
        graph will instead plot per frame.
    '''
    
    #Initialise an empty array to collect data
    framerange = list(range(frame_num))
    ylist_ilastik = np.zeros(len(framerange))
    
    #For each frame, extract the mean of the data type specified by {statnum}
    for i in framerange:
        ylist_ilastik[i] = statistics.mean(data_list[:, stat_num, i])
    
    #Prepare axes
    data_x = times_list
    data_y = ylist_ilastik
    
    #Send data to be plotted
    framelinegraphs(path, data_x, data_y, x_name, y_name, stat_name)
    
def framelinegraphs(path, data_x, data_y, x_name, y_name, stat_name):
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
    x_name : string
        X-axis label.
    y_name : string
        Y-axis label.
    statname : string
        Name of data type to be plotted.

    '''
    fig, ax = plt.subplots(figsize = (20,12))
    plt.plot(data_x,data_y)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    plt.savefig(os.path.join(path, f"iMMPY_DataOut/{stat_name}_all"))
    plt.close()

def Single_bacterium_brightness_plot(path, data_list, stat_num, frame_num, stat_name, x_name, y_name, times_list, error_tolerance_percentage = 0):
    '''
    Prepares data for plotting and evaluates errors. Each bacterium's brightness at each 
    frame will be plotted
    
    Parameters
    ----------
    path : string
        Path of folder containing original images and masks.
    data_list : matrix
        Matrix containing each piece of extracted data.
    stat_num : integer
        Integer that corresponds to the data type being plotted. Used to idex the {datalist}.
    frame_num : integer
        The number of frames in the original masks and images.
    stat_name : string
        Name of data type to be plotted.
    x_name : string
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
    frame_range = list(range(frame_num))
    bac_num = len(data_list[:,0,0])

    ylist_ilastik = np.zeros([bac_num,len(frame_range)])
    
    #prepare for error detection
    error_filter = np.zeros(bac_num)
    num_lost_real_data = 0
    num_problematic_bac = 0
    num_errors = 0
    
    #For each bacterium
    for j in range(bac_num):
        #Prepare
        errors_tracker = 0
        #For each frame
        for k in frame_range:
            this_frame = data_list[j,stat_num ,k]
            #Check to see if there was no data collected (value is stil at the default -0.0001)
            if this_frame == -0.0001:
                #Record this as an error and set its value to zero
                errors_tracker +=1
                this_frame = 0
            #Record value in new matrix
            ylist_ilastik[j,k] = this_frame
        #Check if the bacterium has had too many errors (more than error_tolerance %)
        if errors_tracker > 0.01 * error_tolerance_percentage * len(frame_range):
            #Mark the bacterium as problematic...
            error_filter[j] = 1
            #   ...and record statistics on number of errors, number of problematic bacteria, 
            #   and number of lost non-erroneous pieces of data that belong to those bacteria.
            num_errors += errors_tracker
            num_problematic_bac += 1
            num_lost_real_data += len(frame_range) - errors_tracker
            if errors_tracker == len(frame_range):
                error_filter[j] = 2

    #Prepare axes
    data_x = times_list
    data_y = ylist_ilastik
    
    all_bac_graphs(path, data_x, data_y, x_name, y_name, stat_name, data_list, stat_num,  frame_range, error_filter, removelines = True)
    all_bac_graphs(path, data_x, data_y, x_name, y_name, stat_name + "_errors", data_list, stat_num, frame_range, error_filter)
  
def all_bac_graphs(path, data_x, data_y, x_name, y_name, stat_name, datalist, stat_num, frame_range, error_filter, removelines = False): 
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
    x_name : string
        X-axis label.
    yname : string
        Y-axis label.
    stat_name : string
        Name of data type to be plotted.
    stat_num : integer
        Corresponds to the data type being plotted. Used to idex the {datalist}.
    datalist : matrix
        Matrix containing each piece of extracted data.
    frame_range : list
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
    for j in range(len(data_y[:,0])):
        #If the bacterium isn't problematic, plot it
        if error_filter[j] == 0:
            plt.plot(data_x, data_y[j,:], color="grey")   
        #If the bacterium is problematic, plot it in red,
        #   but only if {Removelines== False}. Else don't plot it at all
        elif error_filter[j] == 1:
            if removelines == False:
                plt.plot(data_x,data_y[j,:], color="r") 
                
    '''step 2: plot the mean brightness line'''
    
    #For each frame, extract mean data
    ylist_mean_b = np.zeros(len(frame_range))
    for i in frame_range:
        statlist= []
        for j in range(len(error_filter)):
            if error_filter[j] == 0:
                statlist.append(datalist[j,stat_num,i])
            if removelines == False:
                if error_filter[j] == 1:
                    statlist.append(datalist[j,stat_num,i])
        ylist_mean_b[i] = np.mean(statlist)
            
    plt.plot(data_x, ylist_mean_b, "b-s", linewidth=4)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    plt.savefig(os.path.join(path, f"iMMPY_DataOut/{i+1}_{stat_name}_all"))
    plt.close()

if __name__ == '__main__':
    main()