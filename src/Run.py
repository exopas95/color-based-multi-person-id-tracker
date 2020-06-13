#!/usr/bin/env python
# -*- coding: utf-8 -*-
from Input import Input
from Output import Output
import sys
import getopt
import Constants
import pygame
import cv2

class Main():

    def __init__(self):
        self.input = Input()
        pygame.init()
        pygame.display.set_mode((Constants.SCREEN_WIDTH, Constants.SCREEN_HEIGHT))
        pygame.display.set_caption("color-based-multi-person-id-tracker")
        screen = pygame.display.get_surface()
        self.output = Output(screen, self.input)

    def run(self):
        person = int(input("Type person's id you want to track: "))
        group_num = int(input("Type number of total dancers: "))
        color_list = {}
        p_x1 = 1050
        p_x2 = 1500
        while True:
            p_x1, p_x2, color_list = self.input.run(person, color_list, group_num, p_x1, p_x2)
            self.ouput.run(p_x1, p_x2)
        self.capture.release()
        self.out.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    options, remainder = getopt.getopt(sys.argv[1:], 's:x:')
    for opt, arg in options:
        if opt in ('-s'):
            song = arg
        elif opt in ('-x'):
            speed = float(arg)
    game = Main()
    game.run()
