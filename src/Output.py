# -*- coding: utf-8 -*-
from Input import Input
import pygame
import Constants


class Output():
    def __init__(self, screen, input):
        self.input = input
        self.screen = screen

        self.sceneClock = pygame.time.Clock()
        self.backgroundColor = (0, 0, 0)

    def renderWebCam(self, p_x1, p_x2):
        frame = self.input.getCurrentFrameAsImage(p_x1, p_x2)
        self.screen.blit(frame, (0, 0))

    def render(self, p_x1, p_x2):
        self.renderWebCam(p_x1, p_x2)

    def run(self, p_x1, p_x2):
        self.screenDelay = self.sceneClock.tick()
        self.screen.fill(self.backgroundColor)
        self.render(p_x1, p_x2)
        pygame.display.flip()
