import cv2
import numpy as np
import cozmo
import asyncio
import math
from cozmo.util import distance_mm, speed_mmps

'''
@class HighFive
Cozmo recognizes your hand and holds up his lift to high-five you, getting happy if you do and annoyed if you don't.
@author - Wizards of Coz
'''

class HighFive():    
    def __init__(self):
        self.cozmo_idle = True                          # Cozmo state - idle or waiting for user to high-five
        self.frames_hand_visible_thresh = 20            # Number of frames Cozmo sees a hand continuously before acknowledging it
        self.frames_wait_for_high_five_thresh = 35      # Number of frames Cozmo waits for a user to high-five
        self.norm_thresh = 0.85                         # Threshold for successful high-five         
        self.cnt1 = 0
        self.cnt2 = 0
        self.blur = 5
        self.kernel = 11

        cv2.namedWindow('Thresholded')
        cv2.createTrackbar('blur', 'Thresholded', 1, 5, self.update_values)
        cv2.createTrackbar('kernel', 'Thresholded', 1, 9, self.update_values)

        self.robot = None
        cozmo.connect(self.run)

    def update_values(self, x):             
        self.blur = 2*(cv2.getTrackbarPos('blur', 'Thresholded')) + 1
        self.kernel = 2*(cv2.getTrackbarPos('kernel', 'Thresholded')) + 1

    
    def wait_for_high_five(self, thresh_img):
        count = 0
        total = 0
        rows, cols = thresh_img.shape
        for i in range (0, rows):
            for j in range (0, cols):
                if thresh_img[i, j] != 0:
                    count += 1
                total += 1
        return count/total
    

    async def see_hand(self):
        await self.robot.play_anim_trigger(cozmo.anim.Triggers.AcknowledgeFaceNamed).wait_for_completed()
        await self.robot.set_head_angle(cozmo.util.Angle(degrees=45), in_parallel = True).wait_for_completed()
        await self.robot.drive_straight(distance_mm(20), speed_mmps(100), in_parallel = True).wait_for_completed()
        await self.robot.set_lift_height(1, accel=100.0, max_speed=100.0, duration=0.0, in_parallel = True).wait_for_completed()

    async def high_five_success(self):
        await self.robot.play_anim_trigger(cozmo.anim.Triggers.ReactToBlockPickupSuccess).wait_for_completed()
        await self.go_idle()

    async def high_five_fail(self):
        await self.robot.play_anim_trigger(cozmo.anim.Triggers.ReactToBlockRetryPickup).wait_for_completed()
        await self.go_idle()

    async def go_idle(self):
        self.cozmo_idle = True
        self.cnt1 = 0
        self.cnt2 = 0
        await self.robot.set_head_angle(cozmo.util.Angle(degrees=45), in_parallel = True).wait_for_completed()
        await self.robot.set_lift_height(0, accel=10.0, max_speed=10.0, duration=0.0, in_parallel = True).wait_for_completed()

    async def on_new_camera_image(self, event, *, image:cozmo.world.CameraImage, **kw):
        img = np.array(image.raw_image)
        cv2.rectangle(img, (90,50), (230,190), (0,0,0), 1)
        crop_img = img[50:190, 90:230]
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur,self.blur), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self.kernel, 2)
        #_, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU )

        cv2.imshow('Thresholded', thresh)
        cv2.imshow('HighFiveCozmo', img)
        k = cv2.waitKey(10)
        if k == 27:
            exit()
                
        if self.cozmo_idle == True:
            image, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            cnt = max(contours, key = lambda x: cv2.contourArea(x))
            hull = cv2.convexHull(cnt)
            hull = cv2.convexHull(cnt, returnPoints = False)
            defects = cv2.convexityDefects(cnt, hull)
            count_defects = 0

            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])
                a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
                if angle <= 90:
                    count_defects += 1

            if count_defects == 1:                      # Number of fingers visible                
                if self.cnt1 < self.frames_hand_visible_thresh:
                    self.cnt1 += 1
                else:
                    self.cozmo_idle = False
                    print("hand seen")
                    self.robot.remove_event_handler(cozmo.world.EvtNewCameraImage, self.event_handler)
                    await self.see_hand()
                    self.event_handler = self.robot.add_event_handler(cozmo.world.EvtNewCameraImage, self.on_new_camera_image)
            else:
                self.cnt1 = 0

        else:
            if self.wait_for_high_five(thresh) > self.norm_thresh:
                print("success")
                self.robot.remove_event_handler(cozmo.world.EvtNewCameraImage, self.event_handler)
                await self.high_five_success()
                self.event_handler = self.robot.add_event_handler(cozmo.world.EvtNewCameraImage, self.on_new_camera_image)
            elif self.cnt2 < self.frames_wait_for_high_five_thresh:
                print("waiting for hi5")
                self.cnt2 += 1
            else:
                print("fail")
                self.robot.remove_event_handler(cozmo.world.EvtNewCameraImage, self.event_handler)
                await self.high_five_fail()
                self.event_handler = self.robot.add_event_handler(cozmo.world.EvtNewCameraImage, self.on_new_camera_image)             

    async def set_up_cozmo(self, conn):
        asyncio.set_event_loop(conn._loop)
        self.robot = await conn.wait_for_robot()
        self.robot.camera.image_stream_enabled = True
        self.event_handler = self.robot.add_event_handler(cozmo.world.EvtNewCameraImage, self.on_new_camera_image)
        await self.robot.set_head_angle(cozmo.util.Angle(degrees=45), in_parallel = True).wait_for_completed()
        await self.robot.set_lift_height(0, accel=10.0, max_speed=10.0, duration=0.0, in_parallel = True).wait_for_completed()

    async def run(self, conn):
        await self.set_up_cozmo(conn)
        while True:
            await asyncio.sleep(0)

if __name__ == '__main__':
    HighFive()
