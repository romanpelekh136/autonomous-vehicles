import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame
import math
import json
import os
from PIL import Image

class CarRacingCustom(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, track_name="track_01"):
        super().__init__()
        self.render_mode = render_mode
        self.num_rays = 21
        
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        low_bounds = np.zeros(self.num_rays + 5, dtype=np.float32)
        low_bounds[-5] = -1.0 
        low_bounds[-3] = -1.0 

        self.observation_space = spaces.Box(
            low=low_bounds, high=np.float32(1.0), dtype=np.float32
        )

        image_path = f"{track_name}.png"
        json_path = f"{track_name}.json"
        
        if not os.path.exists(image_path) or not os.path.exists(json_path):
            raise FileNotFoundError(f"Track files {track_name}.png or {track_name}.json missing.")
            
        pil_img = Image.open(image_path).convert("RGB")
        self.track_array = np.ascontiguousarray(np.array(pil_img).transpose(1, 0, 2))
        self._track_image_path = image_path
        self.track_image = None
        
        with open(json_path, 'r') as f:
            track_data = json.load(f)
            
        self.start_x = track_data["start_position"]["x"]
        self.start_y = track_data["start_position"]["y"]
        self.start_angle = math.radians(track_data["start_position"]["angle"])
        
        self.checkpoints = track_data["checkpoints"]

        self.width = pil_img.width
        self.track_height = pil_img.height
        self.panel_height = 120
        self.height = self.track_height + self.panel_height

        # Маска траси за кольором стартової позиції
        self.tr = self.track_array[int(self.start_x), int(self.start_y), 0]
        self.tg = self.track_array[int(self.start_x), int(self.start_y), 1]
        self.tb = self.track_array[int(self.start_x), int(self.start_y), 2]

        self.track_mask = (self.track_array[:, :, 0] == self.tr) & \
                          (self.track_array[:, :, 1] == self.tg) & \
                          (self.track_array[:, :, 2] == self.tb)

        self.screen = None
        self.clock = None
        self.font = None 

        self.current_checkpoint = 0
        self.laps = 0 
        self.current_action = [0.0, 0.0, 0.0] 
        self.prev_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Параметри фізики
        self.L = 10                    
        self.max_speed = 35.0
        self.max_ray_length = 800
        self.max_steering_speed = 0.15

        self.reset()

    def _get_lidar_data(self):
        """промені на 180° і повертає нормалізовані відстані до країв траси."""

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
        self.cached_lidar = self._get_lidar_data()
        obs = np.concatenate([
            self.cached_lidar,
            [self.current_steering],
            [self.speed / self.max_speed],
            self.prev_action
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
        self.total_checkpoints = 0
        self.steps_count = 0
        self.lap_steps = 0  
        self.stuck_steps = 0 
        self.current_action = [0.0, 0.0, 0.0]
        self.prev_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.cached_lidar = np.zeros(self.num_rays, dtype=np.float32)
        return self._get_observation(), {}

    def step(self, action):
        self.steps_count += 1
        self.lap_steps += 1 
        self.current_action = action 
        
        # зменшення впливу керма на високій швидкості
        speed_damping = np.interp(self.speed, [0, self.max_speed], [1.0, 0.2])
        delta = np.clip(action[0] - self.current_steering, -self.max_steering_speed, self.max_steering_speed)
        self.current_steering += delta
        steering = self.current_steering * speed_damping * (math.pi / 4)
        
        acceleration = action[1]
        brake = action[2]

        self.speed *= 0.98  # опір
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
            progress = self.total_checkpoints / len(self.checkpoints)
            return self._get_observation(), -100.0, True, False, {
                "checkpoints": self.total_checkpoints,
                "progress": progress,
                "crashed": True,
            }

        if self.speed < 1.0 and self.steps_count > 100:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0

        if self.stuck_steps > 120:  # завершення, при застряганні
            progress = self.total_checkpoints / len(self.checkpoints)
            return self._get_observation(), -100.0, True, False, {
                "checkpoints": self.total_checkpoints,
                "progress": progress,
                "crashed": True,
            }

        terminated = False
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
            
        car_dx = math.cos(self.angle)
        car_dy = -math.sin(self.angle)
        dot = car_dx * dir_x + car_dy * dir_y
        safe_dot = max(dot, 0.0)

        # Штраф за простій, бонус за швидкість до чекпоінту
        reward = -0.1
        speed_ratio = self.speed / self.max_speed
        reward += speed_ratio * 2.5 * safe_dot
        forward_ray = self.cached_lidar[self.num_rays // 2]

        # Бонус за газ/гальмо залежно від відстані попереду
        if forward_ray > 0.8:
            reward += action[1] * 0.2
            reward -= action[2] * 0.5
        elif forward_ray < 0.5 and speed_ratio > 0.4:
            reward += action[2] * 0.3

        # Штраф за різкі повороти
        steering_jitter = abs(action[0] - self.prev_action[0])
        reward -= steering_jitter * 0.15

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
            self.total_checkpoints += 1
            reward += 20.0
            if self.current_checkpoint == len(self.checkpoints) - 1:
                self.laps += 1
                reward += 50.0  # бонус за коло
                self.lap_steps = 0
            self.current_checkpoint = (self.current_checkpoint + 1) % len(self.checkpoints)

        self.prev_action = np.copy(action)
        observation = self._get_observation()
        truncated = False
        progress = self.total_checkpoints / len(self.checkpoints)

        return observation, reward, terminated, truncated, {
            "checkpoints": self.total_checkpoints,
            "progress": progress,
            "crashed": False,
        }


    def _ensure_render_resources(self):
        """Ініціалізація шрифтів та зображення траси для рендеру."""
        if not pygame.get_init():
            pygame.init()
        if self.font is None:
            self.font = pygame.font.SysFont('Arial', 20)
        if not hasattr(self, '_lidar_angles') or self._lidar_angles is None:
            self._lidar_angles = np.linspace(-math.pi/2, math.pi/2, self.num_rays)
        if self.track_image is None:
            self.track_image = pygame.image.load(self._track_image_path).convert()

    def render_to_surface(self, surface):
        """Малює поточний кадр  pygame."""
        self._ensure_render_resources()

        surf_w = surface.get_width()
        surf_h = surface.get_height()
        view_h = surf_h - self.panel_height

        surface.fill((0, 0, 0))
        ox = int(surf_w / 2 - self.car_x)
        oy = int(view_h / 2 - self.car_y)
        surface.blit(self.track_image, (ox, oy))

        target_cp = self.checkpoints[self.current_checkpoint]
        pygame.draw.line(surface, (0, 0, 255),
                        (int(target_cp["x1"] + ox), int(target_cp["y1"] + oy)),
                        (int(target_cp["x2"] + ox), int(target_cp["y2"] + oy)), 3)

        cx = int(self.car_x + ox)
        cy = int(self.car_y + oy)
        for i, a in enumerate(self._lidar_angles):
            ray_angle = self.angle + a
            end_x = int(cx + self.cached_lidar[i] * self.max_ray_length * math.cos(ray_angle))
            end_y = int(cy - self.cached_lidar[i] * self.max_ray_length * math.sin(ray_angle))
            pygame.draw.line(surface, (0, 255, 0), (cx, cy), (end_x, end_y), 2)

        car_surface = pygame.Surface((20, 10), pygame.SRCALPHA)
        car_surface.fill((200, 0, 0))
        rotated_car = pygame.transform.rotate(car_surface, math.degrees(self.angle))
        rect = rotated_car.get_rect(center=(cx, cy))
        surface.blit(rotated_car, rect.topleft)

        panel_y = surf_h - self.panel_height
        pygame.draw.rect(surface, (40, 40, 40), (0, panel_y, surf_w, self.panel_height))
        pygame.draw.line(surface, (200, 200, 200), (0, panel_y), (surf_w, panel_y), 3)
        lap_text = self.font.render(f"Laps: {self.laps}", True, (255, 255, 255))
        surface.blit(lap_text, (30, panel_y + 20))
        speed_text = self.font.render(f"Speed: {self.speed:.1f} / {self.max_speed}", True, (255, 255, 255))
        surface.blit(speed_text, (30, panel_y + 60))

        self._draw_bar_on(surface, "Steer", self.current_steering, -1.0, 1.0, 250, panel_y + 40, (0, 200, 255))
        self._draw_bar_on(surface, "Gas", self.current_action[1], 0.0, 1.0, 500, panel_y + 20, (0, 255, 0))
        self._draw_bar_on(surface, "Brake", self.current_action[2], 0.0, 1.0, 500, panel_y + 60, (255, 0, 0))

        return surface

    def _draw_bar_on(self, surface, label, value, min_val, max_val, x, y, color):
        """Малює горизонтальний індикатор на поверхні."""
        if self.font is None:
            return
        text = self.font.render(label, True, (255, 255, 255))
        surface.blit(text, (x, y))
        bar_x = int(x + 80)
        bar_y = int(y + 5)
        bar_w = 120
        bar_h = 15
        pygame.draw.rect(surface, (100, 100, 100), (bar_x, bar_y, bar_w, bar_h), 2)
        val = float(value)
        if min_val < 0:
            center_x = int(bar_x + bar_w / 2)
            fill_w = int(abs(val) * (bar_w / 2))
            if val < 0:
                pygame.draw.rect(surface, color, (center_x - fill_w, bar_y, fill_w, bar_h))
            else:
                pygame.draw.rect(surface, color, (center_x, bar_y, fill_w, bar_h))
            pygame.draw.line(surface, (255, 255, 255), (center_x, bar_y), (center_x, bar_y + bar_h))
        else:
            fill_w = int(val * bar_w)
            pygame.draw.rect(surface, color, (bar_x, bar_y, fill_w, bar_h))

    def render(self):
        if self.render_mode is None:
            return

        if self.screen is None:
            pygame.init()
            self.window_width = min(800, self.width)
            self.window_height = min(600, self.track_height) + self.panel_height
            self.screen = pygame.display.set_mode((self.window_width, self.window_height),
                                                    pygame.DOUBLEBUF | pygame.HWSURFACE)
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 20)
            self._lidar_angles = np.linspace(-math.pi/2, math.pi/2, self.num_rays)
            self.track_image = pygame.image.load(self._track_image_path).convert()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt

        self.render_to_surface(self.screen)

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None