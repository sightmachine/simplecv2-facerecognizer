from simplecv.features.features import Feature


class HaarFeature(Feature):
    """
    **SUMMARY**

    The HaarFeature is a rectangle returned by the FindHaarFeature() function.

    * The x,y coordinates are defined by the center of the bounding rectangle.
    * The classifier property refers to the cascade file used for detection .
    * Points are the clockwise points of the bounding rectangle, starting in
     upper left.

    """
    classifier = ""
    _width = ""
    _height = ""
    neighbors = ''
    feature_name = 'None'

    def __init__(self, i, haarobject, haarclassifier=None, cv2flag=True):
        self.image = i
        if not cv2flag:
            ((x, y, width, height), self.neighbors) = haarobject
        else:
            (x, y, width, height) = haarobject
        at_x = x + width / 2
        at_y = y + height / 2  # set location of feature to middle of rectangle
        points = (
            (x, y), (x + width, y), (x + width, y + height), (x, y + height))

        #set bounding points of the rectangle
        self.classifier = haarclassifier
        if haarclassifier is not None:
            self.feature_name = haarclassifier.get_name()

        super(HaarFeature, self).__init__(i, at_x, at_y, points)

    def draw(self, color=(0, 255, 0), width=1):
        """
        **SUMMARY**

        Draw the bounding rectangle, default color green.

        **PARAMETERS**

        * *color* - An RGB color triplet.
        * *width* - if width is less than zero we draw the feature filled in,
         otherwise we draw the get_contour using the specified width.


        **RETURNS**

        Nothing - this is an inplace operation that modifies the source images
        drawing layer.

        """
        self.image.draw_line(self.points[0], self.points[1], color, width)
        self.image.draw_line(self.points[1], self.points[2], color, width)
        self.image.draw_line(self.points[2], self.points[3], color, width)
        self.image.draw_line(self.points[3], self.points[0], color, width)

    def __getstate__(self):
        sdict = self.__dict__.copy()
        if 'classifier' in sdict:
            del sdict["classifier"]
        return sdict

    def mean_color(self):
        """
        **SUMMARY**

        Find the mean color of the boundary rectangle.

        **RETURNS**

        Returns an  RGB triplet that corresponds to the mean color of the
        feature.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> face = HaarCascade("face.xml")
        >>> faces = img.find_haar_features(face)
        >>> print faces[-1].mean_color()

        """
        crop = self.image[self.points[0][0]:self.points[1][0],
                          self.points[0][1]:self.points[2][1]]
        return crop.mean_color()

    def get_area(self):
        """
        **SUMMARY**

        Returns the area of the feature in pixels.

        **RETURNS**

        The area of the feature in pixels.

        **EXAMPLE**

        >>> img = Image("lenna")
        >>> face = HaarCascade("face.xml")
        >>> faces = img.find_haar_features(face)
        >>> print faces[-1].get_area()

        """
        return self.get_width() * self.get_height()

