class Side:
    def __init__(self, len, angles): #colors_amount, colors,
        #self.colors_amount = colors_amount
        #self.colors = colors # цвета слева на право [color, pix start]
        self.len = len
        self.angles = angles # (l_ang, r_ang) смотрим на сторону с внешней стороны фрагмента


class Piece:
    def __init__(self, sides):
        self.sides = sides