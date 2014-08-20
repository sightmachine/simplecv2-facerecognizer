import os
import tempfile

from nose.tools import assert_equals, assert_is_none, assert_equal, assert_list_equal
from simplecv.factory import Factory
from simplecv.tests.utils import perform_diff
from simplecv.tests import utils

from simplecv_facerecognizer import DATA_DIR
from simplecv_facerecognizer.facerecognizer import FaceRecognizer
from simplecv_facerecognizer.haar_cascade import HaarCascade

utils.standard_path = os.path.join(DATA_DIR, 'test', 'standard')

FACECASCADE = 'face.xml'

testimage = os.path.join(DATA_DIR, "sampleimages/orson_welles.jpg")
testoutput = os.path.join(tempfile.gettempdir(), 'orson_welles_face.jpg')

testneighbor_in = os.path.join(DATA_DIR, "sampleimages/04000.jpg")
testneighbor_out = os.path.join(tempfile.gettempdir(), "sampleimages/04000_face.jpg")


def test_image_recognize_face():
    img = Factory.Image("lenna")
    recognizer = FaceRecognizer()
    faces = img.find_haar_features("face.xml")
    face = faces[0].crop()
    recognizer.load(os.path.join(DATA_DIR, "FaceRecognizer/GenderData.xml"))
    assert_equals(face.recognize_face(recognizer)[0], 0)

    # invalid recognizer
    assert_is_none(img.recognize_face(2))


def test_image_find_and_recognize_faces():
    img = Factory.Image("lenna")
    recognizer = FaceRecognizer()
    recognizer.load(os.path.join(DATA_DIR, "FaceRecognizer/GenderData.xml"))
    assert_equals(img.find_and_recognize_faces(recognizer, "face.xml")[0][1],
                  0)


def test_find_haar_features():
    img = Factory.Image(testimage)
    img1 = img.copy()
    face = HaarCascade("face.xml")  # old HaarCascade
    f = img.find_haar_features(face)
    f2 = img1.find_haar_features("face_cv2.xml")  # new cv2 HaarCascade
    assert len(f) > 0
    assert len(f2) > 0
    f.draw()
    f2.draw()
    f[0].get_width()
    f[0].get_height()
    f[0].length()
    f[0].get_area()

    results = [img, img1]
    name_stem = "test_find_haar_features"
    perform_diff(results, name_stem)

    # incorrect cascade name
    f3 = img.find_haar_features(cascade="incorrect_cascade.xml")
    assert_equals(f3, None)

    # incorrect cascade object
    f4 = img.find_haar_features(cascade=img1)
    assert_equals(f4, None)

    # Empty image
    img2 = Factory.Image((100, 100))
    f5 = img2.find_haar_features("face_cv2.xml")
    assert_equals(f5, None)


def test_list_haar_features():
    features_directory = os.path.join(DATA_DIR, 'HaarCascades')
    features = os.listdir(features_directory)
    assert_equals(features, Factory.Image.list_haar_features())


def test_haarcascade():
    img = Factory.Image(testimage)
    faces = img.find_haar_features(FACECASCADE)

    if faces:
        faces.draw()
        img.save(testoutput)
    else:
        assert False

    cascade = HaarCascade(FACECASCADE, "face_cascade")
    assert_equals(cascade.get_name(), "face_cascade")

    fhandle = os.path.join(DATA_DIR, "HaarCascades", "face.xml")
    assert_equals(cascade.get_fhandle(), fhandle)

    cascade.set_name("eye_cascade")
    assert_equals(cascade.get_name(), "eye_cascade")

    new_fhandle = os.path.join(DATA_DIR, "HaarCascades", "eye.xml")
    cascade.load(new_fhandle)
    assert_equals(cascade.get_fhandle(), new_fhandle)

    emptycascade = HaarCascade()


def test_minneighbors(img_in=testneighbor_in, img_out=testneighbor_out):
    img = Factory.Image(img_in)
    faces = img.find_haar_features(FACECASCADE, min_neighbors=20)
    if faces is not None:
        faces.draw()
        img.save(img_out)
        assert len(faces) <= 1, "Haar Cascade is potentially ignoring the " \
                            "'HIGH' min_neighbors of 20"

def test_anonymize():
    img = Factory.Image(source="lenna")
    anon_img = img.anonymize()

    # provide features
    anon_img1 = img.anonymize(features=["face.xml", "profile.xml"])

    # match both images
    assert_equals(anon_img.get_ndarray().data, anon_img1.get_ndarray().data)

    # transform function
    def transform_blur(img, rect):
        np_array = img.get_ndarray()
        x, y, w, h = rect
        crop_np_array = np_array[y:y+h, x:x+w]
        crop_img = Factory.Image(array=crop_np_array)
        blur_img = crop_img.blur((15, 15))
        blur_np_array = blur_img.get_ndarray()
        np_array[y:y+h, x:x+w] = blur_np_array
        return Factory.Image(array=np_array)

    # apply tranform function
    anon_img2 = img.anonymize(transform=transform_blur)

    perform_diff([anon_img1, anon_img2], "test_anonymize")


"""
def test_facerecognizer():


    images3 = [os.path.join(DATA_DIR, ""../data/sampleimages/fi1.jpg"),
               os.path.join(DATA_DIR, ""../data/sampleimages/fi2.jpg"),
               os.path.join(DATA_DIR, ""../data/sampleimages/fi3.jpg"),
               os.path.join(DATA_DIR, ""../data/sampleimages/fi4.jpg")]


    imgset3 = []

    for img in images3:
        imgset3.append(Image(img))
    label1 = ["female"] * len(imgset3)

    for img in images2:
        imgset2.append(Image(img))
    label2 = ["male"] * len(imgset2)

    imgset = imgset1 + imgset2
    labels = label1 + label2
    imgset[4] = imgset[4].resize(400, 400)
    f.train(imgset, labels)

    for img in images3:
        imgset3.append(Image(img))
    imgset[2].resize(300, 300)
    label = []
    for img in imgset3:
        name, confidence = f.predict(img)
        label.append(name)

    assert_list_equal(["male", "male", "female", "female"], label)
"""


def test_facerecognizer_train():
    images1 = [os.path.join(DATA_DIR, "sampleimages/ff1.jpg"),
               os.path.join(DATA_DIR, "sampleimages/ff2.jpg"),
               os.path.join(DATA_DIR, "sampleimages/ff3.jpg"),
               os.path.join(DATA_DIR, "sampleimages/ff4.jpg"),
               os.path.join(DATA_DIR, "sampleimages/ff5.jpg")]

    images2 = [os.path.join(DATA_DIR, "sampleimages/fm1.jpg"),
               os.path.join(DATA_DIR, "sampleimages/fm2.jpg"),
               os.path.join(DATA_DIR, "sampleimages/fm3.jpg"),
               os.path.join(DATA_DIR, "sampleimages/fm4.jpg"),
               os.path.join(DATA_DIR, "sampleimages/fm5.jpg")]

    images3 = [os.path.join(DATA_DIR, "sampleimages/fi1.jpg"),
               os.path.join(DATA_DIR, "sampleimages/fi2.jpg"),
               os.path.join(DATA_DIR, "sampleimages/fi3.jpg"),
               os.path.join(DATA_DIR, "sampleimages/fi4.jpg")]

    imgset1 = []
    imgset2 = []
    imgset3 = []
    label = []

    for img in images1:
        imgset1.append(Factory.Image(img))
    label1 = ["female"] * len(imgset1)

    for img in images2:
        imgset2.append(Factory.Image(img))
    label2 = ["male"] * len(imgset2)

    imgset = imgset1 + imgset2
    labels = label1 + label2
    imgset[4] = imgset[4].resize(400, 400)

    for img in images3:
        imgset3.append(Factory.Image(img))

    f = FaceRecognizer()
    trained = f.train(
        csvfile=os.path.join(DATA_DIR, "test/standard/test_facerecognizer_train_data.csv"),
        delimiter=",")

    for img in imgset3:
        name, confidence = f.predict(img)
        label.append(name)

    assert_equal(trained, True)
    assert_list_equal(["male", "male", "female", "female"], label)

    fr1 = FaceRecognizer()
    trained = fr1.train(csvfile="no_such_file.csv")
    assert_equal(trained, False)

    fr2 = FaceRecognizer()
    trained = fr2.train(imgset, labels)
    assert_equal(trained, True)

    label = []
    for img in imgset3:
        name, confidence = fr2.predict(img)
        label.append(name)
    assert_list_equal(["male", "male", "female", "female"], label)

    fr3 = FaceRecognizer()
    trained = fr3.train(imgset1, label1)
    assert_equal(trained, False)

    fr4 = FaceRecognizer()
    trained = fr4.train(imgset, label2)
    assert_equal(trained, False)

    prediction = fr4.predict(imgset3[0])
    assert_equal(prediction, None)


def test_facerecognizer_load():
    f = FaceRecognizer()
    trained = f.train(
        csvfile=os.path.join(DATA_DIR, "test/standard/test_facerecognizer_train_data.csv"),
        delimiter=",")

    images3 = [os.path.join(DATA_DIR, "sampleimages/ff1.jpg"),
               os.path.join(DATA_DIR, "sampleimages/ff5.jpg"),
               os.path.join(DATA_DIR, "sampleimages/fm3.jpg"),
               os.path.join(DATA_DIR, "sampleimages/fm4.jpg")]

    imgset3 = []

    for img in images3:
        imgset3.append(Factory.Image(img))

    label = []
    for img in imgset3:
        name, confidence = f.predict(img)
        label.append(name)

    assert_list_equal(['female', 'female', 'male', 'male'], label)

    fr1 = FaceRecognizer()
    trained = fr1.load("no_such_file.xml")
    assert_equal(trained, False)

    prediction = fr1.predict(imgset3[0])
    assert_equal(prediction, None)


def test_facerecognizer_save():
    images1 = [os.path.join(DATA_DIR, "sampleimages/ff1.jpg"),
               os.path.join(DATA_DIR, "sampleimages/ff2.jpg"),
               os.path.join(DATA_DIR, "sampleimages/ff3.jpg"),
               os.path.join(DATA_DIR, "sampleimages/ff4.jpg"),
               os.path.join(DATA_DIR, "sampleimages/ff5.jpg")]

    images2 = [os.path.join(DATA_DIR, "sampleimages/fm1.jpg"),
               os.path.join(DATA_DIR, "sampleimages/fm2.jpg"),
               os.path.join(DATA_DIR, "sampleimages/fm3.jpg"),
               os.path.join(DATA_DIR, "sampleimages/fm4.jpg"),
               os.path.join(DATA_DIR, "sampleimages/fm5.jpg")]

    imgset1 = []
    imgset2 = []

    for img in images1:
        imgset1.append(Factory.Image(img))
    label1 = ["female"] * len(imgset1)

    for img in images2:
        imgset2.append(Factory.Image(img))
    label2 = ["male"] * len(imgset2)

    imgset = imgset1 + imgset2
    labels = label1 + label2

    f = FaceRecognizer()
    trained = f.train(imgset, labels)

    filename = os.path.join(tempfile.gettempdir(), "gendertrain.xml")
    if (trained):
        saved = f.save(filename)
        assert_equal(saved, True)

        if not os.path.exists(os.path.abspath(filename)):
            assert False

        os.remove(filename)
