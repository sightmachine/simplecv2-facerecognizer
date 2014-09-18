import os

from simplecv.base import logger
from simplecv.core.image import image_method, static_image_method
from simplecv.factory import Factory
from simplecv.features.features import FeatureSet
import cv2

from simplecv_facerecognizer import DATA_DIR
from simplecv_facerecognizer.facerecognizer import FaceRecognizer
from simplecv_facerecognizer.haar_cascade import HaarCascade


@image_method
def recognize_face(img, recognizer=None):
    """
    **SUMMARY**

    Find faces in the image using FaceRecognizer and predict their class.

    **PARAMETERS**

    * *recognizer*   - Trained FaceRecognizer object

    **EXAMPLES**

    >>> cam = Camera()
    >>> img = cam.get_image()
    >>> recognizer = FaceRecognizer()
    >>> recognizer.load("training.xml")
    >>> print img.recognize_face(recognizer)
    """
    if not hasattr(cv2, "createFisherFaceRecognizer"):
        logger.warn("OpenCV >= 2.4.4 required to use this.")
        return None

    if not isinstance(recognizer, FaceRecognizer):
        logger.warn("SimpleCV.Features.FaceRecognizer object required.")
        return None

    w, h = recognizer.image_size
    label = recognizer.predict(img.resize(w, h))
    return label


@image_method
def find_and_recognize_faces(img, recognizer, cascade=None):
    """
    **SUMMARY**

    Predict the class of the face in the image using FaceRecognizer.

    **PARAMETERS**

    * *recognizer*  - Trained FaceRecognizer object

    * *cascade*     -haarcascade which would identify the face
                     in the image.

    **EXAMPLES**

    >>> cam = Camera()
    >>> img = cam.get_image()
    >>> recognizer = FaceRecognizer()
    >>> recognizer.load("training.xml")
    >>> feat = img.find_and_recognize_faces(recognizer, "face.xml")
    >>> for feature, label, confidence in feat:
        ... i = feature.crop()
        ... i.draw_text(str(label))
        ... i.show()
    """
    if not hasattr(cv2, "createFisherFaceRecognizer"):
        logger.warn("OpenCV >= 2.4.4 required to use this.")
        return None

    if not isinstance(recognizer, FaceRecognizer):
        logger.warn("SimpleCV.Features.FaceRecognizer object required.")
        return None

    if not cascade:
        cascade = os.path.join(DATA_DIR, 'HaarCascades/face.xml')

    faces = img.find_haar_features(cascade)
    if not faces:
        logger.warn("Faces not found in the image.")
        return None

    ret_val = []
    for face in faces:
        label, confidence = face.crop().recognize_face(recognizer)
        ret_val.append([face, label, confidence])
    return ret_val


@static_image_method
def list_haar_features():
    '''
    This is used to list the built in features available for HaarCascade
    feature detection.  Just run this function as:

    >>> img.list_haar_features()

    Then use one of the file names returned as the input to the
    findHaarFeature() function. So you should get a list, more than likely
    you will see face.xml, to use it then just

    >>> img.find_haar_features('face.xml')
    '''

    features_directory = os.path.join(DATA_DIR, 'HaarCascades')
    features = os.listdir(features_directory)
    return features


# this code is based on code that's based on code from
# http://blog.jozilla.net/2008/06/27/
# fun-with-python-opencv-and-face-detection/
@image_method
def find_haar_features(img, cascade, scale_factor=1.2, min_neighbors=2,
                       use_canny=cv2.cv.CV_HAAR_DO_CANNY_PRUNING,
                       min_size=(20, 20), max_size=(1000, 1000)):
    """
    **SUMMARY**

    A Haar like feature cascase is a really robust way of finding the
    location of a known object. This technique works really well for a few
    specific applications like face, pedestrian, and vehicle detection. It
    is worth noting that this approach **IS NOT A MAGIC BULLET** . Creating
    a cascade file requires a large number of images that have been sorted
    by a human.vIf you want to find Haar Features (useful for face
    detection among other purposes) this will return Haar feature objects
    in a FeatureSet.

    For more information, consult the cv2.CascadeClassifier documentation.

    To see what features are available run img.list_haar_features() or you
    can provide your own haarcascade file if you have one available.

    Note that the cascade parameter can be either a filename, or a
    HaarCascade loaded with cv2.CascadeClassifier(),
    or a SimpleCV HaarCascade object.

    **PARAMETERS**

    * *cascade* - The Haar Cascade file, this can be either the path to a
      cascade file or a HaarCascased SimpleCV object that has already been
      loaded.

    * *scale_factor* - The scaling factor for subsequent rounds of the
      Haar cascade (default 1.2) in terms of a percentage
      (i.e. 1.2 = 20% increase in size)

    * *min_neighbors* - The minimum number of rectangles that makes up an
      object. Ususally detected faces are clustered around the face, this
      is the number of detections in a cluster that we need for detection.
      Higher values here should reduce false positives and decrease false
      negatives.

    * *use-canny* - Whether or not to use Canny pruning to reject areas
     with too many edges (default yes, set to 0 to disable)

    * *min_size* - Minimum window size. By default, it is set to the size
      of samples the classifier has been trained on ((20,20) for face
      detection)

    * *max_size* - Maximum window size. By default, it is set to the size
      of samples the classifier has been trained on ((1000,1000) for face
      detection)

    **RETURNS**

    A feature set of HaarFeatures

    **EXAMPLE**

    >>> faces = HaarCascade(
        ...         "./SimpleCV/data/Features/HaarCascades/face.xml",
        ...         "myFaces")
    >>> cam = Camera()
    >>> while True:
    >>>     f = cam.get_image().find_haar_features(faces)
    >>>     if f is not None:
    >>>          f.show()

    **NOTES**

    OpenCV Docs:
    - http://opencv.willowgarage.com/documentation/python/
      objdetect_cascade_classification.html

    Wikipedia:
    - http://en.wikipedia.org/wiki/Viola-Jones_object_detection_framework
    - http://en.wikipedia.org/wiki/Haar-like_features

    The video on this pages shows how Haar features and cascades work to
    located faces:
    - http://dismagazine.com/dystopia/evolved-lifestyles/8115/
    anti-surveillance-how-to-hide-from-machines/

    """
    if isinstance(cascade, basestring):
        cascade = HaarCascade(cascade)
        if not cascade.get_cascade():
            return None
    elif isinstance(cascade, HaarCascade):
        pass
    else:
        logger.warning('Could not initialize HaarCascade. '
                       'Enter Valid cascade value.')
        return None

    haar_classify = cv2.CascadeClassifier(cascade.get_fhandle())
    objects = haar_classify.detectMultiScale(
        img.to_gray(), scaleFactor=scale_factor,
        minNeighbors=min_neighbors, minSize=min_size,
        flags=use_canny)

    if objects is not None and len(objects) != 0:
        return FeatureSet(
            [Factory.HaarFeature(img, o, cascade, True) for o in objects])

    return None


@image_method
def anonymize(img, block_size=10, features=None, transform=None):
    """
    **SUMMARY**

    Anonymize, for additional privacy to images.

    **PARAMETERS**

    * *features* - A list with the Haar like feature cascades that should
       be matched.
    * *block_size* - The size of the blocks for the pixelize function.
    * *transform* - A function, to be applied to the regions matched
      instead of pixelize.
    * This function must take two arguments: the image and the region
      it'll be applied to,
    * as in region = (x, y, width, height).

    **RETURNS**

    Returns the image with matching regions pixelated.

    **EXAMPLE**

    >>> img = Image("lenna")
    >>> anonymous = img.anonymize()
    >>> anonymous.show()

    >>> def my_function(img, region):
    >>>     x, y, width, height = region
    >>>     img = img.crop(x, y, width, height)
    >>>     return img
    >>>
    >>>img = Image("lenna")
    >>>transformed = img.anonymize(transform = my_function)

    """

    regions = []

    if features is None:
        regions.append(img.find_haar_features("face.xml"))
        regions.append(img.find_haar_features("profile.xml"))
    else:
        for feature in features:
            regions.append(img.find_haar_features(feature))

    print regions
    found = [f for f in regions if f is not None]

    img = img.copy()

    if found:
        for feature_set in found:
            for region in feature_set:
                rect = (region.top_left_corner[0],
                        region.top_left_corner[1],
                        region.width, region.height)
                if transform is None:
                    img = img.pixelize(block_size=block_size, region=rect)
                else:
                    img = transform(img, rect)
    return img