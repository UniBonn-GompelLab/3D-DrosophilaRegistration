import io
import os
import shutil
import glob
import re
import PySimpleGUI as sg
from PIL import Image
import pandas as pd
import numpy as np
import ast
from datetime import datetime
import cv2
from ..registration.TPS import TPSwarping
from skimage.filters import gaussian
from skimage.segmentation import active_contour

file_types_dfs = [("CSV (*.csv)", "*.csv"),("All files (*.*)", "*.*")]

file_types_images = [("tiff (*.tiff)", "*.tif"), ("All files (*.*)", "*.*")]

image_quality_choiches = ["undefined", "good", "fair", "poor", "bad"]

df_files_name = "images_dataframe.csv"

df_landmarks_name = "landmarks_dataframe.csv"

df_model_name = "model_dataframe.csv"

df_channels_name = "extra_channels_dataframe.csv"

ref_image_name = "reference_image.tif"


def update_image_fields(im_index, image, df_files, window, graph_canvas_width):
    """
    Function used to update all the fields related to the current image 
    when the image is changed:
    
    Parameters
    ----------
    im_index : int
        index of the current image.
    image : numpy array
    
    df_files : pandas Dataframe
    
    window : PySimplegui window
    
    graph_canvas_width: int
        width of the graph element of the main window

    Returns
    -------
    None.

    """

    filename = df_files.loc[im_index,"file name"]
    image_quality = df_files.loc[im_index,"image quality"]
    image_notes = df_files.loc[im_index,"notes"]
    image_annotated = df_files.loc[im_index,"annotated"]
    window["-CURRENT-IMAGE-"].update(value=filename)
    window["-IMAGE-QUALITY-"].update(value=image_quality)
    window["-IMAGE-NOTES-"].update(value=image_notes)
    window["-IMAGE-ANNOTATED-"].update(value=image_annotated)
    update_image(image, window, graph_canvas_width)
    
    return


def update_image(image, window, canvas_width):
    """
    Function used to show an image on the graph element of a window
    
    Parameters
    ----------
    image : numpy array
        image to show on the window graph element.
    window : PySimplegui window
        window with a graph object.
    canvas_width : int
        width of the graph object

    Returns
    -------
    None.

    """
    
    width = image.width
    height = image.height
    scaling_factor = canvas_width/width
    new_width = canvas_width
    new_height = int(height*scaling_factor)
    
    image = image.resize((new_width, new_height))

    bio = io.BytesIO()
    image.save(bio, format="PNG")
    window['-GRAPH-'].erase()
    window['-GRAPH-'].set_size((new_width, new_height))
    window['-GRAPH-'].change_coordinates((0,0), (width, height), )
    window['-GRAPH-'].draw_image(data=bio.getvalue(), location=(0,height))
    return

def open_image(image_path, normalize=True):
    """
    Opens the image file, converts it into 8 bit, and optionally, normalizes it 
    
    Parameters
    ----------
    image_path : str
        path to the image.
    normalize : bool, optional.

    Returns
    -------
    a Pillow image.

    """
    image = Image.open(image_path)
    image = np.asarray(image)
    image = convert_image_to8bit(image, normalize)
    image = Image.fromarray(image)
    return image

def update_landmarks_preview(image_path, window, canvas_width, normalize=True):
    """
    Function used to update the image in the landmarks preview element of the main window.
    It opens the reference image file, converts it into 8 bit, normalizes it and resizes it.
    
    Parameters
    ----------
    image_path : str
        path to the image.
    window : PySimplegui window
        main window with the -LANDMARK-PREVIEW- element.
    canvas_width : int
        width of the -LANDMARK-PREVIEW- graph element.
    normalize : bool, optional
        wether the image should be normalized before visualization. The default is True.

    Returns
    -------
    None.

    """
    image = Image.open(image_path)
    image = np.asarray(image)
    image = convert_image_to8bit(image, normalize)
    image = Image.fromarray(image)
    
    width = image.width
    height = image.height
    scaling_factor = canvas_width/width
    new_width = canvas_width
    new_height = int(height*scaling_factor)
    
    image = image.resize((new_width, new_height))

    bio = io.BytesIO()
    image.save(bio, format="PNG")
    window['-LANDMARKS-PREVIEW-'].erase()
    window['-LANDMARKS-PREVIEW-'].set_size((new_width, new_height))
    window['-LANDMARKS-PREVIEW-'].change_coordinates((0,0), (width, height), )
    window['-LANDMARKS-PREVIEW-'].draw_image(data=bio.getvalue(), location=(0,height))
    return

def refresh_gui_with_new_image(shared, df_files, df_model, df_landmarks, main_window, landmarks_window):
    """
    Parameters
    ----------
    shared : dictionary
        contains data shared across windows, definition is in the main function.
    df_files : pandas DataFrame
        dataframe containing the paths to images of current project.
    df_model : pandas DataFrame
        dataframe containing the names and positions of landmarks of current project.
    df_landmarks : pandas DataFrame
        dataframe containing the positions of landmarks and additional notes for
        all images of current project.
    main_window : PySimplegui window
        main window of the GUI.
    landmarks_window : PySimplegui window
        landmark selection window of the GUI.

    Returns
    -------
    shared : dictionary
        updated data shared across windows.
    landmarks_window: PySimplegui window
        resfreshed landmark selection window of the GUI.
    """
    
    # updated current image, raw_image and current file:
    shared['curr_image'] = open_image(df_files.loc[shared['im_index'],"full path"], normalize=shared['normalize'])
    shared['raw_image'] = shared['curr_image']
    shared['curr_file'] = df_files.loc[shared['im_index'],"file name"]
    shared['pt_size'] = shared['curr_image'].width / 80
    
    # update all the fields related to the image (image qualty, notes, etc..)
    update_image_fields(shared['im_index'], shared['curr_image'], df_files, main_window, shared['graph_width'])
    
    # refresh the landmarks window, if present
    if landmarks_window:
        location = landmarks_window.CurrentLocation()
        temp_window = make_landmarks_window(df_model, df_landmarks, shared['curr_file'], location = location)
        landmarks_window.Close()
        landmarks_window = temp_window
        
    # else, create a new one:
    else:
        landmarks_window = make_landmarks_window(df_model, df_landmarks, shared['curr_file'])
    
    # update the preview of the landmarks: 
    update_landmarks_preview(os.path.join(shared['proj_folder'], ref_image_name), main_window, 300)
    
    # remove selection of the current landmark
    shared['curr_landmark'] = None
    
    # update the progress bar
    update_progress_bar(df_files, main_window)
    
    # remove focus from any object
    try:
        x = main_window.FindElementWithFocus()
        x.block_focus()
    except:
        pass
    
    # place the focus on the main window:
    main_window.TKroot.focus_force() 
    
    return shared, landmarks_window


def convert_image_to8bit(image, normalize=False):
    """
    Helper function to convert an image in 8 bit format and, optionally, normalize it
    
    Parameters
    ----------
    image : numpy array
    normalize : bool, optional
        wether the image shoudl be normalized. The default is False.

    Returns
    -------
    image : numpy array

    """
    if normalize:
        image = image - np.min(image)
        image = image/np.max(image)
        image = 255*image
    else:
        max_val = np.iinfo(image.dtype).max
        image = image/max_val
        image = 255*image
        
    image = np.uint8(image)
    return image

def convert_image_coordinates_to_graph(x, y, im_width, im_height):
    """
    
    Function used to convert image coordinates in graph coordinates.
    The y coordinate in a PySimplegui Graph element is inverted compared to 
    the usual definition for images.
    
    Parameters
    ----------
    x : int
    y : int
    im_width : int
        width of the image.
    im_height : int
        height of the image.

    Returns
    -------
    [x,y] list
        graph coordinates.

    """
    y = im_height - y
    return [x, y]

def convert_graph_coordinates_to_image(x, y,im_width, im_height):
    """
    Function used to convert graph coordinates in image coordinates.
    The y coordinate in a PySimplegui Graph element is inverted compared to 
    the usual definition for images.
    
    Parameters
    ----------
    x : int
    y : int
    im_width : int
        width of the image.
    im_height : int
        height of the image.

    Returns
    -------
    [x,y] list
        image coordinates.

    """
    y = im_height - y
    return [x, y]
 
def update_progress_bar(df_files, window):
    """
    Function used to update the progress bar in the main window of the GUI 
    and print a message that shows how many images have been annotated.

    Parameters
    ----------
    df_files : TYPE
        DESCRIPTION.
    window : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    n_not_annotated = len(df_files[df_files['annotated']=="No"].index)
    n_annotated = len(df_files[df_files['annotated']=="Yes"].index)
    progress = 100*(n_annotated/(n_annotated+n_not_annotated))
    window["-PROGRESS-"].update(current_count = progress)
    window["-PRINT-"].update(str(n_annotated)+" annotated images out of "+str(n_not_annotated+n_annotated) )
    return   

def rebin(img, binning):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    width = int(img.shape[1] / binning)
    height = int(img.shape[0] / binning)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized


def enhance_edges(img, binning, smoothing):
    img = rebin(img, binning)
    # filtered_image = difference_of_gaussians(img, 2, 10)
    # could use the following to avoid relying on scikit-image 0.19
    filtered_image = gaussian(img, 10) - gaussian(img, 2)
    filtered_image = (-filtered_image)*(filtered_image<0).astype(np.uint8)
    filtered_image = 1 -filtered_image
    edges = (filtered_image-np.min(filtered_image))/(np.max(filtered_image)-np.min(filtered_image))
    edges = gaussian(edges, smoothing, preserve_range=False)
    return edges

#Issue with point spacing : if point spacing > lenght between the two points -> error
def snake_contour(img, p1_x, p1_y, p2_x, p2_y, N = None, points_spacing = 30, binning=5, smoothing=2, alpha=0.1):
    distance = np.sqrt((p1_x-p2_x)**2+(p1_y-p2_y)**2)
    n_points = N or int(distance/points_spacing)

    r = np.linspace(p1_y, p2_y, n_points)/binning
    c = np.linspace(p1_x, p2_x, n_points)/binning
        
    init = np.array([r, c]).T
    img = enhance_edges(img, binning, smoothing)
    snake = active_contour(img,
                   init, boundary_condition='fixed', coordinates='rc', 
                   alpha=alpha, beta=0.1, w_line=-5, w_edge=0, gamma= 0.1)
    return snake*binning
    

def snake(img, df_model, df_lines):
   
    LM = []
    N=[]
    for i in range(len(df_lines["Lmk1"])):
        lm1 = ast.literal_eval(df_model.loc[df_model["name"]==df_lines["Lmk1"][i], "target"].values[0])
        lm2 = ast.literal_eval(df_model.loc[df_model["name"]==df_lines["Lmk2"][i], "target"].values[0])
        alpha = (df_lines.loc[df_lines["Lmk1"][i]==df_model["name"], "alpha"])
        alpha = alpha.iloc[0]
        snk = snake_contour(img,lm1[0],lm1[1],lm2[0],lm2[1],alpha=alpha)
        LM.extend(snk)
        N.append(len(snk))
    return LM,N


def create_new_project():
    """
    Function used to create a new project.
    It opens a new graphical window and collects from the user the info required
    for the creation of a new project: 
        
        - the project location
        - a folder of raw images
        - a reference image
        - a model file containig the definitions of the landmarks to be used by
          the project
          
    Finally, it creates all the new project files in the target folder.
          
    """
    
    # GUI - Define a new window to collect input:
    layout = [[sg.Text("Project name: ", size=(20, 1)),
               sg.Input(size=(25,8), enable_events=True,  key='-NEW-PROJECT-NAME-')],
              
              [sg.Text('Project location: ', size=(20, 1)), 
               sg.Input(size=(25,8), enable_events=True, key='-NEW-PROJECT-FOLDER-'),
               sg.FolderBrowse()],
              
              [sg.Text("Image files extension: ", size=(20, 1)),
               sg.Input(size=(25,8), enable_events=True,  key='-IMAGE-EXTENSION-')],
              
              [sg.Text('Raw images location: ', size=(20, 1)), 
               sg.Input(size=(25,8), enable_events=True, key='-NEW-IMAGES-FOLDER-'),
               sg.FolderBrowse()], 
              
              [sg.Text("Reference image: ", size=(20, 1)),
               sg.Input(size=(25,8), enable_events=True,  key="-NEW-REF-IMAGE-"),
               sg.FileBrowse(file_types=file_types_images)], 
              
              [sg.Text("Model file: ", size=(20, 1)),
               sg.Input(size=(25,8), enable_events=True,  key="-NEW-MODEL-FILE-"),
               sg.FileBrowse(file_types=file_types_dfs)],
              
              [sg.Button("Create the project: ", size = (20,1), key="-CREATE-PROJECT-")],
              
              [sg.Frame("Dialog box: ", layout = [[sg.Text("", key="-DIALOG-", size=(50, 10))]])]
              
              ]
    
    new_project_window = sg.Window("Create New Project", layout, modal=True)
    choice = None
    
    while True:
        event, values = new_project_window.read()

        if event == '-CREATE-PROJECT-':
            ## Create the folder
            parent_folder = values['-NEW-PROJECT-FOLDER-']
            project_name  = values['-NEW-PROJECT-NAME-']
            
            project_folder = os.path.join(parent_folder, project_name)
            
            dialog_box = new_project_window["-DIALOG-"]
            
            dialog_box = new_project_window["-DIALOG-"]
            
            if os.path.exists(project_folder):
                dialog_box.update(value=dialog_box.get()+'\n - The project folder already exists.')
                break
            else:
                os.mkdir(project_folder)
                dialog_box.update(value=dialog_box.get()+'\n - New project folder has been created.')
                
            extension = values['-IMAGE-EXTENSION-']
            images_folder = values['-NEW-IMAGES-FOLDER-']
            
            temp_path = os.path.join(images_folder,r"**")
            temp_path = os.path.join(temp_path,r"*."+extension)
    
            image_full_paths = glob.glob(temp_path, recursive=True)
            image_names = [os.path.split(path)[1] for path in image_full_paths]
            
            dialog_box.update(value=dialog_box.get()+'\n - '+str(len(image_names))+' images found.')
            
            df_files = pd.DataFrame({'file name':image_names,'full path':image_full_paths})
            df_files["image quality"] = "undefined"
            df_files["notes"] = "none"
            df_files["annotated"] = "No"
            
            df_files = df_files.drop_duplicates(subset='file name', keep="first")
            
            df_files.to_csv(os.path.join(project_folder, df_files_name))
            df_files.to_csv(os.path.join(project_folder, df_files_name), index=False)
            dialog_box.update(value=dialog_box.get()+'\n - Dataframe with file names created.')
            
            reference_image_path = values['-NEW-REF-IMAGE-']
            
            try:
                new_ref_image_path = os.path.join(project_folder, ref_image_name)
                shutil.copy(reference_image_path, new_ref_image_path)
                dialog_box.update(value=dialog_box.get()+'\n - Reference image copied in the project folder.')
            except:
                dialog_box.update(value=dialog_box.get()+'\n ***ERROR*** \n - "Error occurred while copying reference image file."')
            
            new_model_path = values['-NEW-MODEL-FILE-']
            df_model = pd.read_csv(new_model_path)
            df_model.to_csv(os.path.join(project_folder, df_model_name))
            df_model.to_csv(os.path.join(project_folder, df_model_name), index=False)
            dialog_box.update(value=dialog_box.get()+'\n - "Dataframe with model information copied in the project folder.')
            
            try:
                landmark_names = df_model['name'].values
                df_landmarks = df_files[['file name']].copy()
                for landmark in landmark_names:
                    df_landmarks[landmark] = np.nan
                
                df_landmarks.to_csv(os.path.join(project_folder, df_landmarks_name))
                df_landmarks.to_csv(os.path.join(project_folder, df_landmarks_name), index=False)
                dialog_box.update(value=dialog_box.get()+'\n - "Dataframe for landmarks coordinates created.')
            except:
                dialog_box.update(value=dialog_box.get()+'\n ***ERROR*** \n - "Problem in the creation of the landmarks dataframe.')
 
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        
    new_project_window.close()
    
    return


def create_channels_dataframe(df_files, folders, ref_channel, dialog_box):
    """
    Function used to generate to locate the image files of additional channels 
    corresponding to each image of the reference channels in the project.
    The function considers all the files included in the df_files dataframe and 
    look in a list of folders for files with a matching name except for the 
    channel label.
    
    Parameters
    ----------
    df_files : pandas dataframe
        .
    folders : str
        a string containing multiple folders, separated by newlines.
    ref_channel : str
        a string defining which part of the filename corresponds to a channel label.
    dialog_box : pysimplegui textbox
        the dialoge box of the registration window, for printing messages/errors.

    Returns
    -------
    df_channels : pandas dataframe
        a pandas dataframe containing the filename and channel name of files 
        corresponding to additional channels of each image file in the original 
        df_files dataframe.

    """
    # Make sure folders is a list of folder paths:
    folders = folders.split("\n")
    file_names = df_files["file name"].unique()
    
    # Initialize df_channels:
    df_channels = pd.DataFrame(columns=['file name', 'extra channel name', 'full path'])
    skipped_files = False
    for file_name in file_names:
        try:
            file_name_part1, file_name_part2 = file_name.split(ref_channel)
            file_pattern = os.path.join("**", file_name_part1+r"*"+file_name_part2)
            
            for folder in folders:
                new_file_names = glob.glob(os.path.join(folder, file_pattern), recursive=True)
                for new_file_name in new_file_names:
                    new_file_name_tail = os.path.split(new_file_name)[1]
                    channel = new_file_name_tail.split(file_name_part1)[1].split(file_name_part2)[0]
                    if new_file_name_tail  == file_name:
                        pass
                    else:
                        # append new row to df_channels:
                        row_data = [[file_name, channel, new_file_name]]
                        row_columns = ['file name', 'extra channel name', 'full path']
                        row = pd.DataFrame(row_data, columns=row_columns)
                        df_channels = pd.concat([row, df_channels])
                        pass
        except:
            skipped_files = True
            pass
    if skipped_files:
        dialog_box.update(value="WARNING: some files in your project will be skipped during the registration. \n Please double check the results. ")
    
    return df_channels


def create_registration_window(shared, df_landmarks, df_model, df_files):
    """
    Function used to register the images in the project.
    It starts a new window where the user can specify where to save
    the registered images, if the transformation should be applied to additional
    channels and where the images of additional channels are located.
    An option for binning the original images to a smaller size is also provided.
          
    """

    # GUI - Define a new window to collect input:
    layout = [
              [sg.Text('Target folder for registered images: ', size=(30, 1)), 
               sg.Input(size=(30,1), enable_events=True, key='-REGISTERED-IMAGES-FOLDER-'),
               sg.FolderBrowse()],
              [sg.Text('Snake model file: ', size=(30, 1)), 
               sg.Input(size=(30,1), enable_events=True, key='-SNAKE-MODEL-'),
               sg.FileBrowse()],
              [sg.Text('Image resolution of registered images (%):',size=(38,1)),
              sg.Slider(orientation ='horizontal', key='-REGISTRATION-RESOLUTION-', range=(1,100),default_value=100)],
              [sg.Checkbox('Apply the registration to additional channels?', key="-MULTI-CHANNEL-", default=False, enable_events=True)],
              [sg.Text("List all folders where to search images corresponding to additional channels:", visible=False, key="-TEXT-CH-")],
              [sg.Multiline(enter_submits=False,  size = (60,5), key='-EXTRA-CHANNELS-FOLDERS-', autoscroll=True, visible=False, do_not_clear=True)],
              [sg.Text('Insert the name of the reference channel: ', key="-TEXT-CH2-", size=(60, 1), visible=False)], 
              [sg.Input(size=(62, 1), enable_events=True, key='-REFERENCE-CHANNEL-', visible=False)],
              [sg.Button("Start Registration ", size = (20,2), key="-REGISTRATION-SAVE-")],
              [sg.Text("Progress:")],
              [sg.ProgressBar(max_value=100, size=(50,10), key= "-PROGRESS-")],
              [sg.Frame("Dialog box: ", layout = [[sg.Text("", key="-DIALOG-", size=(60, 5))]])]
              ]
    
    registration_window = sg.Window("Registration of the annotated images", layout, modal=True)

    dialog_box = registration_window["-DIALOG-"]
    df_snake = None
    
    while True:
        event, values = registration_window.read()
        
        if values["-MULTI-CHANNEL-"] == True:
            registration_window.Element('-TEXT-CH-').Update(visible=True)
            registration_window.Element('-EXTRA-CHANNELS-FOLDERS-').Update(visible=True)
            registration_window.Element('-TEXT-CH2-').Update(visible=True)
            registration_window.Element('-REFERENCE-CHANNEL-').Update(visible=True)

        if values["-MULTI-CHANNEL-"] == False:
            registration_window.Element('-TEXT-CH-').Update(visible=False)
            registration_window.Element('-EXTRA-CHANNELS-FOLDERS-').Update(visible=False)
            registration_window.Element('-TEXT-CH2-').Update(visible=False)
            registration_window.Element('-REFERENCE-CHANNEL-').Update(visible=False)
      
        if event == '-SNAKE-MODEL-':
            df_snake = pd.read_csv(values['-SNAKE-MODEL-'])              

        if event == '-REGISTRATION-SAVE-':
            # Index for loading bar:
            loading_bar_i=0   
            dialog_box.update(value='Registration started, this may take a while..')
            # Getting reference landmarks
            c_dst=[]
            landmarks_list = df_model["name"].values
            
            for landmark in shared['list_landmarks']:
                [x,y] = ast.literal_eval(df_model.loc[df_model["name"]==landmark, "target"].values[0])
                c_dst.append([x,y])

            # Getting snake landmarks for reference
            if df_snake is not None:
                df_snake["N_points"] = 0
                for index, row in df_snake.iterrows():
                    # get the positions of the two landmarks in target image:
                    lmk1_name, lmk2_name  = row["Lmk1"], row["Lmk2"]
                    lmk1_pos = ast.literal_eval(df_model.loc[df_model["name"]==lmk1_name, "target"].values[0])
                    lmk2_pos = ast.literal_eval(df_model.loc[df_model["name"]==lmk2_name, "target"].values[0])
                    alpha = row["alpha"]
                    smoothing = row["smoothing"]
                    w_line = row["w_line"]
                    ref_image = cv2.imread(os.path.join(shared['proj_folder'], ref_image_name),  cv2.IMREAD_ANYDEPTH)
                    snk = snake_contour(ref_image, lmk1_pos[1], lmk1_pos[0], lmk2_pos[1], lmk2_pos[0], alpha, smoothing, w_line)
                    df_snake.loc[index,"N_points"] = len(snk)            
                    c_dst.extend(snk[1:-1].tolist())
    
            c_dst = np.reshape(c_dst,(len(c_dst),2))
            shape_dst = np.asarray(shared['ref_image'].size)
            c_dst = c_dst/shape_dst
            
            # Get the images and their landmarks
            file_names = df_files["file name"].unique()
            file_count = len(file_names)
            
            # If the multiple channels option is selected, create the dataframe 
            # with the file names of corresponding images.
            if values["-MULTI-CHANNEL-"]:
                folders = values['-EXTRA-CHANNELS-FOLDERS-']
                ref_channel = values['-REFERENCE-CHANNEL-']
                df_channels = create_channels_dataframe(df_files, folders, ref_channel, dialog_box)

            # create a dataframe to store the info about the registered images, 
            # which will also be saved in the destination folder:
                
            df_info = pd.DataFrame(columns=['file name', 'channel', 'image quality', 'notes'])
            
            
            # Start looping through the images to register:
            for file_name in file_names:
                # Refresh the dialog box:
                dialog_box.update(value='Image registration in progress')
                
                # Open the source image:
                file_path = df_files.loc[df_files["file name"] == file_name, "full path"].values[0]
                img = cv2.imread(file_path,  cv2.IMREAD_ANYDEPTH)
                shape_src = np.asarray(img.shape)
                
                # Get image landmarks
                c_src=[]
 
                for LM in landmarks_list:
                    try:
                        LM_position = df_landmarks.loc[df_landmarks["file name"]==file_name, LM].values[0]
                        c_src.append(ast.literal_eval(LM_position))
                    except:
                        pass
                
                # Check if some landmarks are missing, and skip the image
                if len(c_src) != len(landmarks_list):
                    loading_bar_i+=1
                    continue 
                
                # Get snake image landmarks
                if df_snake is not None:
                    # binning:
                    binning = np.max([1, np.round( max(shape_src)/max(shape_dst) )])
                    for index, row in df_snake.iterrows():
                        # get the positions of the two landmarks in target image:
                        lmk1_name = row["Lmk1"]
                        lmk2_name = row["Lmk2"]
                        lmk1_pos = ast.literal_eval(df_landmarks.loc[df_landmarks["file name"]==file_name, lmk1_name ].values[0])
                        lmk2_pos = ast.literal_eval(df_landmarks.loc[df_landmarks["file name"]==file_name, lmk2_name ].values[0])
                        alpha = row["alpha"]
                        smoothing = row["smoothing"]
                        w_line = row["w_line"]
                        N = row["N_points"]
                        snk = snake_contour(img, lmk1_pos[1], lmk1_pos[0], lmk2_pos[1], lmk2_pos[0], alpha, smoothing, w_line, N=N, binning=binning)
                        c_src.extend(snk[1:-1].tolist())

                np.reshape(c_src,(len(c_src),2))
                
        
       
                c_src = c_src/np.asarray([img.shape[1], img.shape[0]])
                
                # Apply tps, the aspect ratio of the warped image is the same as the target image but with the resolution
                # of the source image.
                warped_shape = tuple( (shape_dst*max(shape_src)/max(shape_dst)).astype(int) )
                warped = TPSwarping(img, c_src, c_dst, warped_shape)
                
                # Resize the image according to the slider value
                size = warped.shape*np.array([values['-REGISTRATION-RESOLUTION-']/100,values['-REGISTRATION-RESOLUTION-']/100])
                size = [int(x) for x in size]
                warped = cv2.resize(warped,(size[1],size[0]))
                
                # Save the registered image
                destination_path = os.path.join(values['-REGISTERED-IMAGES-FOLDER-'], file_name)
                cv2.imwrite(destination_path, warped)
                
                image_quality = df_files.loc[df_files["file name"] == file_name, "image quality"].values[0]
                notes = df_files.loc[df_files["file name"] == file_name, "notes"].values[0]
                df_info_row_data = [[file_name, "registration", image_quality, notes]]
                df_info_row_columns = ['file name', 'channel', 'image quality', 'notes']
                df_info_row = pd.DataFrame(df_info_row_data, columns=df_info_row_columns)
                df_info = pd.concat([df_info_row, df_info])
                dialog_box.update(value=dialog_box.get()+'\n - ' + file_name + ' has been registered')
                
                # If the multiple channels option is selected, apply the thin-plate-spline
                # transformation to the additional images.
                if values["-MULTI-CHANNEL-"]:
                    temp_df = df_channels.loc[df_channels["file name"] == file_name]
                    channels = temp_df['extra channel name'].unique()
                    for ch in channels:
                         ch_file_path = temp_df.loc[temp_df['extra channel name']==ch,"full path"].values[0]
                         ch_file_name = os.path.basename(ch_file_path)
                         ch_img = cv2.imread(ch_file_path,  cv2.IMREAD_ANYDEPTH)
                         ch_warped = TPSwarping(ch_img, c_src, c_dst, warped_shape)
                         ch_warped = cv2.resize(ch_warped,(size[1],size[0]))
                         ch_destination_path = os.path.join(values['-REGISTERED-IMAGES-FOLDER-'], ch_file_name)
                         cv2.imwrite(ch_destination_path, ch_warped)
                         df_info_row_data = [[ch_file_name, ch, image_quality, notes]]
                         df_info_row = pd.DataFrame(df_info_row_data, columns=df_info_row_columns)
                         df_info = pd.concat([df_info_row, df_info])
                         dialog_box.update(value=dialog_box.get()+'\n - ' + ch_file_name + ' has been registered')

                # update the loading bar
                loading_bar_i+=1
                registration_window["-PROGRESS-"].update((loading_bar_i/file_count)*100)

            dialog_box.update(value='\n - All of the images have been registered')
            df_info = df_info.reset_index(drop=True)
            df_info.to_csv(os.path.join(values['-REGISTERED-IMAGES-FOLDER-'],'dataframe_info.csv'))
            
        if event == "Exit" or event == sg.WIN_CLOSED:
            registration_window.close()
            break

    registration_window.close()
    
    return

def merge_projects():
    """
    Function used to merge two existing projects.
    It opens a new graphical window where the user sleect the paths to the two
    proojects to merge and the path where to create the new project.

    Finally, it creates all the new project files in the target folder.
          
    """
    
    # GUI - Define a new window to collect input:
    layout = [[sg.Text("Project name: ", size=(20, 1)),
               sg.Input(size=(25,8), enable_events=True,  key='-NEW-PROJECT-NAME-')],
              
              [sg.Text('Project location: ', size=(20, 1)), 
               sg.Input(size=(25,8), enable_events=True, key='-NEW-PROJECT-FOLDER-'),
               sg.FolderBrowse()],
              
              [sg.Text('Location of Project 1: ', size=(20, 1)), 
               sg.Input(size=(25,8), enable_events=True, key='-PROJECT-FOLDER-1-'),
               sg.FolderBrowse()], 
              
              [sg.Text('Location of Project 2: ', size=(20, 1)), 
               sg.Input(size=(25,8), enable_events=True, key='-PROJECT-FOLDER-2-'),
               sg.FolderBrowse()], 

              [sg.Button("Create the project: ", size = (20,1), key="-CREATE-PROJECT-")],
              
              [sg.Frame("Dialog box: ", layout = [[sg.Text("", key="-DIALOG-", size=(50, 10))]])]
              
              ]
    
    merge_projects_window = sg.Window("Merge Projects", layout, modal=True)
    choice = None
    

    
    while True:
        event, values = merge_projects_window.read()

        if event == '-CREATE-PROJECT-':
            
            ## Create the folder
            parent_folder = values['-NEW-PROJECT-FOLDER-']
            project_name  = values['-NEW-PROJECT-NAME-']
            
            project_folder = os.path.join(parent_folder, project_name)
            
            dialog_box = merge_projects_window["-DIALOG-"]
            
            if os.path.exists(project_folder):
                dialog_box.update(value=dialog_box.get()+'\n - The project folder already exists.')
                break
            else:
                os.mkdir(project_folder)
                dialog_box.update(value=dialog_box.get()+'\n - New project folder has been created.')
                
            folder_1 = values['-PROJECT-FOLDER-1-']
            
            try:
                df_files_1      = pd.read_csv( os.path.join(folder_1, df_files_name) )
                df_landmarks_1  = pd.read_csv( os.path.join(folder_1, df_landmarks_name) )
                df_model_1      = pd.read_csv( os.path.join(folder_1, df_model_name) )
                reference_image_path = os.path.join(folder_1, ref_image_name)
            except:
                dialog_box.update(value=dialog_box.get()+'\n - Problem opening project files of the the first project.')
                    
            folder_2 = values['-PROJECT-FOLDER-2-']
            
            try:
            
                df_files_2      = pd.read_csv( os.path.join(folder_2, df_files_name) )
                df_landmarks_2  = pd.read_csv( os.path.join(folder_2, df_landmarks_name) )
                df_model_2      = pd.read_csv( os.path.join(folder_2, df_model_name) )
            except:
                dialog_box.update(value=dialog_box.get()+'\n - Problem opening project files of the the second project.')
               
            if set(df_landmarks_1.columns) == set(df_landmarks_2.columns):
                
                df_files = pd.concat([df_files_1, df_files_2])
                df_files = df_files.drop_duplicates(subset='file name', keep="first")
                
                df_landmarks = pd.concat([df_landmarks_1, df_landmarks_2])
                df_landmarks = df_landmarks.drop_duplicates(subset='file name', keep="first")     

                try:
                    new_ref_image_path = os.path.join(project_folder, ref_image_name)
                    shutil.copy(reference_image_path, new_ref_image_path)
                    df_files.to_csv(os.path.join(project_folder, df_files_name))
                    df_landmarks.to_csv(os.path.join(project_folder, df_landmarks_name))
                    df_model_1.to_csv(os.path.join(project_folder, df_model_name))
                    dialog_box.update(value=dialog_box.get()+'\n - Merged files saved in the destination folder.')
                    
                except:
                    dialog_box.update(value=dialog_box.get()+'\n ***ERROR*** \n - "Error occurred while merging the files."')
                
            else:
                dialog_box.update(value=dialog_box.get()+'\n - The projects are based on different models. Can not merge.')
            
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        
    merge_projects_window.close()
    
    return

def add_new_images(shared, df_files, df_landmarks, df_model):
    """
    Function used to add new images to an existing project
          
    """
    
    # GUI - Define a new window to collect input:
    layout = [             
              [sg.Text("New image files extension: ", size=(20, 1)),
               sg.Input(size=(25,8), enable_events=True,  key='-IMAGE-EXTENSION-')],
              
              [sg.Text('New images location: ', size=(20, 1)), 
               sg.Input(size=(25,8), enable_events=True, key='-NEW-IMAGES-FOLDER-'),
               sg.FolderBrowse()], 
              
              [sg.Button("Update the project: ", size = (20,1), key="-UPDATE-PROJECT-")],
              
              [sg.Frame("Dialog box: ", layout = [[sg.Text("", key="-DIALOG-", size=(50, 10))]])]
              
              ]
    
    add_images_window = sg.Window("Add new images", layout, modal=True)
    choice = None
    
    while True:
        event, values = add_images_window.read()

        if event == '-UPDATE-PROJECT-':

            project_folder = shared['proj_folder']
        
            extension = values['-IMAGE-EXTENSION-']
            images_folder = values['-NEW-IMAGES-FOLDER-']
            
            temp_path = os.path.join(images_folder,r"**")
            temp_path = os.path.join(temp_path,r"*."+extension)
    
            image_full_paths = glob.glob(temp_path, recursive=True)
            image_names = [os.path.split(path)[1] for path in image_full_paths]
            
            dialog_box = add_images_window["-DIALOG-"]
            dialog_box.update(value=dialog_box.get()+'\n - '+str(len(image_names))+' images found.')
            
            temp_df_files = pd.DataFrame({'file name':image_names,'full path':image_full_paths})
            temp_df_files["image quality"] = "undefined"
            temp_df_files["notes"] = "none"
            temp_df_files["annotated"] = "No"
            
            df_files = pd.concat([df_files, temp_df_files])
            df_files = df_files.drop_duplicates(subset='file name', keep="first")
            df_files = df_files.reset_index(drop=True)
            df_files.to_csv(os.path.join(project_folder, df_files_name))
            dialog_box.update(value=dialog_box.get()+'\n - Dataframe with file names updated.')

            landmark_names = df_model['name'].values
            temp_df_landmarks = df_files[['file name']].copy()
            for landmark in landmark_names:
                temp_df_landmarks[landmark] = np.nan
                
            df_landmarks = pd.concat([df_landmarks, temp_df_files])
            df_landmarks = df_landmarks.reset_index(drop=True)
            df_landmarks.to_csv(os.path.join(project_folder, df_landmarks_name))
            dialog_box.update(value=dialog_box.get()+'\n - "Dataframe for landmarks coordinates updated.')

        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        
    add_images_window.close()
    
    return df_files, df_landmarks

def select_image(shared, df_files):
    """
    Function used to jump to a specific imag ein the current project.
    It creates a pop-up window and allows to search among all the file names 
    included in the current project.
    It returns an updated 'shared' dictionary where the index of the current image 
    has been updated to point to the selected file name.
          
    """
    try:
        names = df_files["file name"].values
    except:
        names = []
    
    layout = [  [sg.Text('Search an image:')],
            [sg.Input(do_not_clear=True, size=(20,1),enable_events=True, key='_INPUT_')],
            [sg.Listbox(names, size=(20,10), enable_events=True, key='_LIST_')],
            [sg.Button('Exit')]]

    select_image_window = sg.Window('Search').Layout(layout)
    # Event Loop
    while True:
        event, values = select_image_window.Read()
        if event is None or event == 'Exit':                # always check for closed window
            break
        if values['_INPUT_'] != '':                         # if a keystroke entered in search field
            search = values['_INPUT_']
            new_values = [x for x in names if search in x] 
            select_image_window.Element('_LIST_').Update(new_values)
        else:
            select_image_window.Element('_LIST_').Update(names)
        if event == '_LIST_' and len(values['_LIST_']):    
            chosen_file = values['_LIST_'][0]
            sg.Popup('Selected ', chosen_file)
            index = 0
            try:
                index = int( df_files[df_files["file name"]==chosen_file].index[0] )
            except:
                pass
            shared['im_index'] = index
            
    select_image_window.Close()
    return shared

def make_main_window(size, graph_canvas_width):
    """
    Function used to create the main window, conatins the definition of the layout
    of the GUI.
    
    Parameters
    ----------
    size : (int, int)
        width and height of the main window.
        
    graph_canvas_width : int, optional
        width of the graph elements where images are visualized, in pixels. The default is 700px.


    Returns
    -------
    None.

    """
    
    # --------------------------------- Define Layout ---------------------------------

    selection_frame = [[sg.Text('Open existing project: ', size=(20, 1)), 
                        sg.Input(size=(20,1), enable_events=True, key='-PROJECT-FOLDER-'),
                        sg.FolderBrowse(size=(20,1)),
                        sg.Button("Load selected project", size=(20,1), key='-LOAD-PROJECT-')],
                       [sg.VPush()],
                       [sg.Button("Registration", size = (20,1), key="-REGISTRATION-"),
                        sg.Button("Create New project", size = (18,1), key="-NEW-PROJECT-", pad=((135,0),(0,0))),
                        sg.Button("Add images to project", size = (20,1), key="-NEW-IMAGES-"),
                        sg.Button("Merge projects", size = (20,1), key="-MERGE-PROJECTS-")]]
    
    image_column = [[sg.Text("Image:", size=(10, 1)), 
                     sg.Text("", key="-CURRENT-IMAGE-", size=(35, 1)),
                     sg.Button("Select image", key="-SELECT-IMAGE-")],
                    [sg.Checkbox('Normalize the image preview', key="-NORMALIZATION-", default=True, enable_events=True),
                     sg.Text("Change Brightness:", size=(20, 1)), 
                     sg.Slider(range=(1, 100), key = "-BRIGHTNESS-", orientation='h', size=(15, 20), default_value=100, enable_events=True,  disable_number_display=True)],
                    [sg.Graph(  canvas_size=(graph_canvas_width, graph_canvas_width),
                                graph_bottom_left=(0, 0),
                                graph_top_right=(graph_canvas_width, graph_canvas_width),
                                key="-GRAPH-",
                                enable_events=True,
                                background_color='white',
                                drag_submits=False)]]
                
    
    annotation_column = [[sg.Button("Next"), sg.Button("Previous"), sg.Button("Next not annotated")],
                         [sg.Text("Image quality:")],
                         [sg.Combo(values=image_quality_choiches, size=(35,10), enable_events=True, key='-IMAGE-QUALITY-')],
                         [sg.Text("Additional notes: ", size=(20, 1))], 
                         [sg.Input(size=(15, 1), key="-IMAGE-NOTES-")],
                         [sg.Text("Image annotated: ", size=(20, 1))], 
                         [sg.Combo(values=["No", "Yes"], size=(35,10), enable_events=True, key='-IMAGE-ANNOTATED-')],
                         [sg.Text("Annotation completion [%]: ")],
                         [sg.ProgressBar(max_value=100, size=(30,10), key= "-PROGRESS-")],
                         [sg.Text("Landmarks model preview: ")],
                         [sg.Graph(canvas_size=(300, 300),
                                graph_bottom_left=(0, 0),
                                graph_top_right=(300, 300),
                                key="-LANDMARKS-PREVIEW-",
                                enable_events=False,
                                drag_submits=False,
                                background_color='white')],
                         [sg.Checkbox('Draw Line', default = False, font = 'Arial 18', key='-LINE-',  enable_events=True)],
                         ]
       
    annotation_frame = [[sg.Column(image_column), sg.Column(annotation_column)]]
    
    communication_window = [[sg.Text("", key="-PRINT-", size=(100, 10))]]
    
    layout = [[sg.Frame("Select project: ", layout = selection_frame)],
              [sg.Frame("Annotate images: ", layout = annotation_frame)],
              [sg.Button("Save changes to the project", key="-SAVE-")],
              [sg.Frame("Messages: ", layout = communication_window)]
              ]
    
    # --------------------------------- Create Window ---------------------------------
    
    return sg.Window("Image Annotation Tool", layout, size=size, finalize=True, return_keyboard_events=True)


def make_landmarks_window(model_df, landmarks_df, current_filename, location = (1200,100), size = (300,1200)):
    """
    Function used to create the extra window with buttons used to select the 
    various landmarks. The window is created only when a project is open and is 
    recreated every time the current image is changed.
    
    Parameters
    ----------
    model_df : dataframe
    
    landmarks_df : dataframe
    
    current_file_name : str
    name of the current image
    
    location : (int, int), optional
    Position of the window. The default is (1200,100).
    
    size : (int, int), optional
    Size of the window. The default is (300,900).

    Returns
    -------
    None.

    """
    landmarks_list = model_df["name"].values
    landmarks_buttons_colors = []
    
    for LM in landmarks_list:
        
        LM_position = landmarks_df.loc[landmarks_df["file name"]==current_filename, LM].values[0]
        
        if LM_position != LM_position: # check if LM_position is np.nan
            landmarks_buttons_colors.append("FloralWhite") 
        else:
            landmarks_buttons_colors.append("SteelBlue3")     
        
    layout = [[sg.Text('Select landmark: ', size=(20, 1))],
              *[[sg.Button(LM, size=(20,1), key = LM, button_color = ("black", landmarks_buttons_colors[i])),] for i, LM in enumerate(landmarks_list)],
              [sg.Button("Show all landmarks", size = (20,1), key = "-SHOW-ALL-")],
              [sg.Button("Delete current Landmark", size = (20,1), key = "-DELETE_LDMK-", button_color = ("black", "orange"))],
             ]
    
    return sg.Window("Landmark selection", layout, size=size, finalize=True, location = location)

def mouse_click_callback(event, window):
    """
    Function that can be bound to a click event in a window. It checks if the click
    happened inside a window element and, if not, rises a -WINDOW-CLICK- event.
    It is used to define the behaviour of the program when the user clicks on 
    the background of the window.
    
    Parameters
    ----------
    event : str
        event name.
    window : PySimpleGUI window
        window where the event took place.

    Returns
    -------
    None.

    """
    x, y = window.TKroot.winfo_pointerxy()
    try:
        widget = window.TKroot.winfo_containing(x, y)
    except:
        widget = None
    for element in window.element_list():
        element_widget = element.Widget
        # if the mouse click happened inside any element of the window, simply 
        # return nothing:
        if widget  == element_widget:
            return
    # otherwise, rise a "-WINDOW-CLICK-" event
    window.write_event_value('-WINDOW-CLICK-', (x, y))
    return