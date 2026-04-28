import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame
import math
import json
import os

class CarRacingCustom(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None, track_name="track_01"):
        super().__init__()
        self.render_mode = render_mode
        self.num_rays = 11 
        
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        low_bounds = np.zeros(self.num_rays + 2, dtype=np.float32)
        low_bounds[-2] = -1.0 # Кермо може бути від'ємним

        self.observation_space = spaces.Box(
            low=low_bounds, high=1.0, dtype=np.float32
        )

        image_path = f"{track_name}.png"
        json_path = f"{track_name}.json"
        
        if not os.path.exists(image_path) or not os.path.exists(json_path):
            raise FileNotFoundError(f"Track files {track_name}.png or {track_name}.json missing.")
            
        self.track_image = pygame.image.load(image_path)
        self.track_array = np.ascontiguousarray(pygame.surfarray.array3d(self.track_image))
        
        with open(json_path, 'r') as f:
            track_data = json.load(f)
            
        self.start_x = track_data["start_position"]["x"]
        self.start_y = track_data["start_position"]["y"]
        self.start_angle = math.radians(track_data["start_position"]["angle"])
        
        self.checkpoints = track_data["checkpoints"]

        self.width = self.track_image.get_width()
        self.track_height = self.track_image.get_height()
        self.panel_height = 120
        self.height = self.track_height + self.panel_height 

        self.track_color = self.track_image.get_at((int(self.start_x), int(self.start_y)))
        self.tr, self.tg, self.tb = self.track_color[0], self.track_color[1], self.track_color[2]

        self.track_mask = (self.track_array[:, :, 0] == self.tr) & \
                          (self.track_array[:, :, 1] == self.tg) & \
                          (self.track_array[:, :, 2] == self.tb)

        self.screen = None
        self.clock = None
        self.font = None 

        self.current_checkpoint = 0
        self.laps = 0 
        self.current_action = [0.0, 0.0, 0.0] 

        self.L = 10
        self.max_speed = 35.0
        self.max_ray_length = 350
        self.max_steering_speed = 0.15

        self.reset()

    def _get_lidar_data(self):
        ray_angles = self.angle + np.linspace(-math.pi/2, math.pi/2, self.num_rays)
        
        dx = np.cos(ray_angles)
        dy = -np.sin(ray_angles)
        
        steps = np.arange(1, self.max_ray_length + 1)[:, np.newaxis]
        
        ray_x = self.car_x + steps * dx
        ray_y = self.car_y + steps * dy
        
        ix = ray_x.astype(int)
        iy = ray_y.astype(int)
        
        valid = (ix >= 0) & (ix < self.width) & (iy >= 0) & (iy < self.track_height)
        
        collision_mask = np.ones((self.max_ray_length, self.num_rays), dtype=bool)
        
        if np.any(valid):
            is_track = self.track_mask[ix[valid], iy[valid]]
            collision_mask[valid] = ~is_track
            
        collision_mask[-1, :] = True
        
        hit_indices = np.argmax(collision_mask, axis=0)
        distances = (hit_indices + 1) / self.max_ray_length
        
        return distances.astype(np.float32)

    def _get_observation(self):
        lidar_data = self._get_lidar_data()
        obs = np.concatenate([
            lidar_data.astype(np.float32),
            [self.current_steering],
            [self.speed / self.max_speed]
        ])
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.car_x = self.start_x
        self.car_y = self.start_y
        self.angle = self.start_angle
        self.speed = 0.0
        self.current_steering = 0.0
        self.current_checkpoint = 0
        self.laps = 0
        self.steps_count = 0
        self.current_action = [0.0, 0.0, 0.0]
        
        return self._get_observation(), {}

    def step(self, action):
        self.steps_count += 1
        self.current_action = action 
        
        old_steering = self.current_steering
        
        speed_factor = 1.0 - (self.speed / self.max_speed) * 0.5 
        target_steer = np.clip(action[0], -speed_factor, speed_factor)
        
        delta = np.clip(target_steer - self.current_steering, -self.max_steering_speed, self.max_steering_speed)
        self.current_steering += delta
        
        steering = self.current_steering * (math.pi / 4)
        
        acceleration = action[1]
        brake = action[2]

        self.speed *= 0.98
        self.speed += acceleration * 1.5
        self.speed -= brake * 3.0
        self.speed = np.clip(self.speed, 0, self.max_speed)

        if self.speed > 0:
            self.angle += (self.speed / self.L) * math.tan(steering)
        
        self.car_x += self.speed * math.cos(self.angle)
        self.car_y -= self.speed * math.sin(self.angle)

        car_pos = (int(self.car_x), int(self.car_y))
        off_track = False
        
        if car_pos[0] < 0 or car_pos[0] >= self.width or car_pos[1] < 0 or car_pos[1] >= self.track_height:
            off_track = True
        elif not self.track_mask[car_pos[0], car_pos[1]]:
            off_track = True
                
        if off_track:
            return self._get_observation(), -100.0, True, False, {}

        terminated = False
        
        # Розрахунок вектору напрямку до наступного чекпоінту
        target_cp = self.checkpoints[self.current_checkpoint]
        cp_cx = (target_cp["x1"] + target_cp["x2"]) / 2.0
        cp_cy = (target_cp["y1"] + target_cp["y2"]) / 2.0
        
        dx = cp_cx - self.car_x
        dy = cp_cy - self.car_y
        dist = math.hypot(dx, dy)
        
        if dist > 0:
            dir_x = dx / dist
            dir_y = dy / dist
        else:
            dir_x, dir_y = 0.0, 0.0
            
        # Вектор напрямку самої машини
        car_dx = math.cos(self.angle)
        car_dy = -math.sin(self.angle)
        
        # Скалярний добуток (від 1.0 до -1.0)
        dot = car_dx * dir_x + car_dy * dir_y
        
        # Базовий штраф за час (щоб не стояла на місці)
        reward = -0.1 
        
        # Штраф за спробу зламати обмежувач керма (використовуємо old_steering)
        steering_diff = abs(action[0] - old_steering)
        if steering_diff > self.max_steering_speed:
            reward -= 0.05 * (self.speed / self.max_speed)
        
        # Бонус за швидкість у правильному напрямку
        speed_bonus = (self.speed / self.max_speed) * 0.8
        reward += speed_bonus * dot
        
        # Мікро-штраф за відхилення керма (стимул їхати прямо)
        reward -= abs(action[0]) * 0.005

        prev_pos = (self.car_x - self.speed * math.cos(self.angle), 
                    self.car_y + self.speed * math.sin(self.angle))
        curr_pos = (self.car_x, self.car_y)
        
        target_cp = self.checkpoints[self.current_checkpoint]
        cp_line = ((target_cp["x1"], target_cp["y1"]), (target_cp["x2"], target_cp["y2"]))
        
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
            
        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
            
        if intersect(prev_pos, curr_pos, cp_line[0], cp_line[1]):
            reward += 20.0 
            if self.current_checkpoint == len(self.checkpoints) - 1:
                self.laps += 1
            self.current_checkpoint = (self.current_checkpoint + 1) % len(self.checkpoints)

        observation = self._get_observation()
        truncated = False

        return observation, reward, terminated, truncated, {}

    # Допоміжна функція для малювання шкал
    def _draw_bar(self, label, value, min_val, max_val, x, y, color):
        text = self.font.render(label, True, (255, 255, 255))
        self.screen.blit(text, (x, y))
        
        bar_x = int(x + 80)
        bar_y = int(y + 5)
        bar_w = 120
        bar_h = 15
        
        # Рамка
        pygame.draw.rect(self.screen, (100, 100, 100), (bar_x, bar_y, bar_w, bar_h), 2)
        
        # Конвертуємо numpy-тип у звичайне число
        val = float(value)
        
        # Заповнення
        if min_val < 0: # Для керма
            center_x = int(bar_x + bar_w / 2)
            fill_w = int(abs(val) * (bar_w / 2))
            if val < 0:
                pygame.draw.rect(self.screen, color, (center_x - fill_w, bar_y, fill_w, bar_h))
            else:
                pygame.draw.rect(self.screen, color, (center_x, bar_y, fill_w, bar_h))
            pygame.draw.line(self.screen, (255, 255, 255), (center_x, bar_y), (center_x, bar_y + bar_h))
        else: # Для газу і гальма
            fill_w = int(val * bar_w)
            pygame.draw.rect(self.screen, color, (bar_x, bar_y, fill_w, bar_h))

    def render(self):
        if self.screen is None:
            pygame.init()
            self.window_width = min(1280, self.width)
            self.window_height = min(720, self.track_height) + self.panel_height
            self.screen = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 20) 
            self.cam_x = 0.0
            self.cam_y = 0.0
            self.zoom = 1.0
            self.panning = False
            self.pan_start_mouse = (0, 0)
            self.pan_start_cam = (0, 0)
            self.render_surface = pygame.Surface((self.width, self.track_height))

        # Process events first for responsiveness
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            elif event.type == pygame.VIDEORESIZE:
                self.window_width, self.window_height = event.w, event.h
                self.screen = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 2: # Middle click
                    self.panning = True
                    self.pan_start_mouse = pygame.mouse.get_pos()
                    self.pan_start_cam = (self.cam_x, self.cam_y)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 2:
                    self.panning = False
            elif event.type == pygame.MOUSEMOTION:
                if self.panning:
                    mouse_pos = pygame.mouse.get_pos()
                    self.cam_x = self.pan_start_cam[0] + (mouse_pos[0] - self.pan_start_mouse[0])
                    self.cam_y = self.pan_start_cam[1] + (mouse_pos[1] - self.pan_start_mouse[1])
            elif event.type == pygame.MOUSEWHEEL:
                mouse_pos = pygame.mouse.get_pos()
                if mouse_pos[1] < self.window_height - self.panel_height:
                    old_world_x = (mouse_pos[0] - self.cam_x) / self.zoom
                    old_world_y = (mouse_pos[1] - self.cam_y) / self.zoom
                    if event.y > 0:
                        self.zoom = min(5.0, self.zoom * 1.1)
                    elif event.y < 0:
                        self.zoom = max(0.1, self.zoom / 1.1)
                    self.cam_x = mouse_pos[0] - old_world_x * self.zoom
                    self.cam_y = mouse_pos[1] - old_world_y * self.zoom

        # --- VIEWPORT RENDERING (тільки видима частина карти) ---
        self.screen.fill((0, 0, 0))
        
        # Автоматичне фокусування камери
        if not self.panning:
            self.cam_x = self.window_width / 2 - self.car_x * self.zoom
            self.cam_y = (self.window_height - self.panel_height) / 2 - self.car_y * self.zoom

        # Визначаємо видиму область карти (viewport)
        vp_x1 = max(0, int(-self.cam_x / self.zoom))
        vp_y1 = max(0, int(-self.cam_y / self.zoom))
        vp_x2 = min(self.width, int((-self.cam_x + self.window_width) / self.zoom))
        vp_y2 = min(self.track_height, int((-self.cam_y + self.window_height - self.panel_height) / self.zoom))
        
        vp_w = max(1, vp_x2 - vp_x1)
        vp_h = max(1, vp_y2 - vp_y1)
        
        # Малюємо тільки видиму частину треку
        viewport_surf = pygame.Surface((vp_w, vp_h))
        viewport_surf.blit(self.track_image, (0, 0), area=(vp_x1, vp_y1, vp_w, vp_h))
        
        # Чекпоінт (зі зміщенням у viewport)
        target_cp = self.checkpoints[self.current_checkpoint]
        pygame.draw.line(viewport_surf, (0, 0, 255), 
                        (target_cp["x1"] - vp_x1, target_cp["y1"] - vp_y1),
                        (target_cp["x2"] - vp_x1, target_cp["y2"] - vp_y1), 3)
        
        # Лідар
        angles = np.linspace(-math.pi/2, math.pi/2, self.num_rays)
        obs = self._get_lidar_data()
        car_vx = self.car_x - vp_x1
        car_vy = self.car_y - vp_y1
        for i, a in enumerate(angles):
            ray_angle = self.angle + a
            end_x = car_vx + obs[i] * self.max_ray_length * math.cos(ray_angle)
            end_y = car_vy - obs[i] * self.max_ray_length * math.sin(ray_angle)
            pygame.draw.line(viewport_surf, (0, 255, 0), (int(car_vx), int(car_vy)), (int(end_x), int(end_y)), 2)
        
        # Машина
        car_length = 20
        car_width = 10
        car_surface = pygame.Surface((car_length, car_width), pygame.SRCALPHA)
        car_surface.fill((200, 0, 0))
        rotated_car = pygame.transform.rotate(car_surface, math.degrees(self.angle))
        rect = rotated_car.get_rect(center=(car_vx, car_vy))
        viewport_surf.blit(rotated_car, rect.topleft)
        
        # Масштабуємо тільки viewport (маленьку частину, не всю карту!)
        if self.zoom != 1.0:
            viewport_surf = pygame.transform.scale(viewport_surf, 
                            (int(vp_w * self.zoom), int(vp_h * self.zoom)))
        
        screen_x = int(self.cam_x + vp_x1 * self.zoom)
        screen_y = int(self.cam_y + vp_y1 * self.zoom)
        self.screen.blit(viewport_surf, (screen_x, screen_y))

        # --- ПАНЕЛЬ ПРИЛАДІВ ---
        panel_y = self.window_height - self.panel_height
        pygame.draw.rect(self.screen, (40, 40, 40), (0, panel_y, self.window_width, self.panel_height))
        pygame.draw.line(self.screen, (200, 200, 200), (0, panel_y), (self.window_width, panel_y), 3)

        # Текстова інформація
        lap_text = self.font.render(f"Laps: {self.laps}", True, (255, 255, 255))
        self.screen.blit(lap_text, (30, panel_y + 20))
        
        speed_text = self.font.render(f"Speed: {self.speed:.1f} / {self.max_speed}", True, (255, 255, 255))
        self.screen.blit(speed_text, (30, panel_y + 60))

        # Шкали дій
        self._draw_bar("Steer", self.current_steering, -1.0, 1.0, 250, panel_y + 40, (0, 200, 255))
        self._draw_bar("Gas", self.current_action[1], 0.0, 1.0, 500, panel_y + 20, (0, 255, 0))
        self._draw_bar("Brake", self.current_action[2], 0.0, 1.0, 500, panel_y + 60, (255, 0, 0))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None