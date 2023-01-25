Image annotation GUI
==============================

What Is This?
-------------

A graphical interface to annotate and place landmarks on images, built with the PySimpleGUI TKinter framework. The GUI can be used to place landmarks on a set of images based on any model. 


How To Run This
---------------

1. Download the package and extract it into a local directory.
2. cd into the root directory where setup.py is located 
3. Run 'python setup.py install'
4. You can now 'import image_registration' from any python terminal and run 'image_registration.gui.start()' to start the graphical interface.

How To Use the GUI
------------------

#### 1. Create a new project

A new project consists of a set of images and a model. A model consists of a reference image and a .csv file containing the names and positions of the landmarks on the reference image, an example model is provided.
Clicking on "Create new project" will open a new window that guides the user in the creation of a new project folder and will automatically generate the relevant files. ( Note: the original raw images for a project are not copied in the project folder but remain at their original location. )


#### 2. Open a project
Either type, or click on "Browse", to provide the path to a project folder, then click on "Load selected project". The first image in the dataset is visualized in the main graph element of the window, while the reference model image is visualized in the smaller graph element. A new window is opened, which allows to select the landmarks present in the current project.


#### 3. Navigate through the dataset and place landmarks and annotations
The buttons "Next", "Previous" and "Next not annotated" are used to navigate through the dataset. Selecting a landmark in the landmark window will highlight the position of that landmark in the reference image, as well as on the current image, if it has been already defined. To define or modify the position of the landmark on the current image just click on the image.
Two drop down menus in the main window are used to assign a label to the image based on its quality (good/fair/poor/bad) and to confirm if the annotation of the image is complete ("Yes/No").
How many images in the dataset have been already annotated is shown by a loading bar as well as by a message in the "Messages" box. 
The "Draw Line" checkbox allows drawing lines on the current image instead of moving/defining landmarks; if selected, two clicks on different positions of the image will create a line connecting them and extending to the edges of the image.

#### 4. Keyboard shortcuts
1. 'Ctrl + 2' loops through the landmarks.
2. 'Ctrl + b' or 'f', 'p', 'b' set the image quality to good, fair, poor and bad.
3. 'Ctrl+a' sets the Image Annotated field to 'Yes'
4. 'Ctrl+s' save changes to the project.
