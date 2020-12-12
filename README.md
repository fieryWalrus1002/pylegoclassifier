# pylegoclassifier
python modules for machine vision matlab integration

pylegoclassifier.py contains the following classes, each with their own methods. 
1. ImageS

# accessing this module in matlab
Reference: https://www.mathworks.com/help/matlab/matlab_external/create-object-from-python-class.html

From the MATLAB command prompt, add the current folder that has the module pylegoclassifier.py in it to the Python search path.

if count(py.sys.path,'') == 0
    insert(py.sys.path,int32(0),'');
end



# there is a dummy method in the image segmentation class for testing purposes. It looks like this.
def dummy_method(self, a):
        if type(a) is np.ndarray:
            result = "object is a numpy.ndarray"
            return result
        else:
            result = "object is a " + str(type(a)) + "and I'm gonna have a hard time with that"
            return result
            
if you call it, you'll get a string returned with the data type of the object. We can use this to test integration with matlab.

# maybe this works?

1. create a numpy n-dimensional array from the matlab matrix and pass the image from matlab into python
img = py.numpy.array([2,3]) :

2. instantiate the image segmentation class as an object in matlab
imgseg = py.pylegoclassifier.ImageSegmenter()

3. call the dummy method, and give it 
however_you_print_in_matlab(imgseg.dummy_method(img))

4. you should get a string response with the type of the object. I'm aiming for the ndarray but its interesting to know what comes back.


# stuff that python needs to have installed on the system in order to work:
'c:\yourPythonDirectory\python.exe -m pip install numpy matplotlib opencv-contrib-python scipy skimage pandas sklearn'

That command will install all of the modules currently needed to run this.
