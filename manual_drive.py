import pygame
import math
from custom_env import CarRacingCustom

env = CarRacingCustom(render_mode="human", track_name="track_01")
obs, _ = env.reset()

running = True
while running:
    env.render()
    
    keys = pygame.key.get_pressed()
    
    steer = 0.0
    current_steer = 0.0

    # замість миттєвого стрибка — накопичуй
    if keys[pygame.K_LEFT]:
        current_steer = max(current_steer - 0.05, -1.0)
    elif keys[pygame.K_RIGHT]:
        current_steer = min(current_steer + 0.05, 1.0)
    else:
        current_steer *= 0.85  # повертає до нуля
    
    gas = 1.0 if keys[pygame.K_UP] else 0.0
    brake = 1.0 if keys[pygame.K_DOWN] else 0.0
    
    obs, reward, terminated, truncated, info = env.step([current_steer, gas, brake])    
    if terminated:
        print("Розбився!")
        obs, _ = env.reset()
    
    # Q для виходу
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False

env.close()