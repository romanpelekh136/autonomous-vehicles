import pygame
import json
import sys
import math
import os

def main():
    if len(sys.argv) > 1:
        track_name = sys.argv[1]
    else:
        track_name = "track_02"

    image_path = f"{track_name}.png"
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return

    pygame.init()
    
    try:
        track_image = pygame.image.load(image_path)
    except pygame.error as e:
        print(f"Failed to load image: {e}")
        return

    orig_width = track_image.get_width()
    orig_height = track_image.get_height()
    
    # Set a maximum window size
    screen_width = min(1280, orig_width)
    screen_height = min(720, orig_height)
    
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
    track_image = track_image.convert_alpha()
    scaled_image = track_image  # Make sure to update the scaled image reference
    pygame.display.set_caption(f"Map Editor - {track_name}")
    
    checkpoints = [] # list of (x1, y1, x2, y2)
    start_pos = None # {"x": x, "y": y, "angle": angle}
    
    # Editor state
    current_line_start = None
    setting_start_dir = False
    temp_start_pos = None
    
    # Viewport state
    zoom = 1.0
    cam_x = 0.0
    cam_y = 0.0
    panning = False
    pan_start_mouse = (0, 0)
    pan_start_cam = (0, 0)
    
    scaled_image = track_image
    
    def update_scaled_image():
        nonlocal scaled_image
        if zoom == 1.0:
            scaled_image = track_image
        else:
            scaled_image = pygame.transform.smoothscale(track_image, (int(orig_width * zoom), int(orig_height * zoom)))
            
    def screen_to_world(pos):
        return ((pos[0] - cam_x) / zoom, (pos[1] - cam_y) / zoom)
        
    def world_to_screen(pos):
        return (pos[0] * zoom + cam_x, pos[1] * zoom + cam_y)
    
    font = pygame.font.SysFont(None, 24)
    clock = pygame.time.Clock()
    
    running = True
    while running:
        screen.fill((50, 50, 50))
        
        # Draw image
        screen.blit(scaled_image, (cam_x, cam_y))
        
        mouse_pos = pygame.mouse.get_pos()
        world_mouse = screen_to_world(mouse_pos)
        
        # Draw existing checkpoints
        for i, cp in enumerate(checkpoints):
            p1 = world_to_screen((cp[0], cp[1]))
            p2 = world_to_screen((cp[2], cp[3]))
            pygame.draw.line(screen, (0, 0, 255), p1, p2, max(1, int(3 * zoom)))
            # Label
            txt = font.render(str(i), True, (255, 255, 255))
            screen.blit(txt, ((p1[0] + p2[0])//2, (p1[1] + p2[1])//2))
            
        # Draw current drawing line
        if current_line_start:
            p1 = world_to_screen(current_line_start)
            pygame.draw.line(screen, (100, 100, 255), p1, mouse_pos, max(1, int(2 * zoom)))
            
        # Draw start pos
        if start_pos:
            sp = world_to_screen((start_pos["x"], start_pos["y"]))
            pygame.draw.circle(screen, (255, 0, 0), (int(sp[0]), int(sp[1])), max(2, int(5 * zoom)))
            rad = math.radians(start_pos["angle"])
            end_x = start_pos["x"] + 30 * math.cos(rad)
            end_y = start_pos["y"] - 30 * math.sin(rad)
            ep = world_to_screen((end_x, end_y))
            pygame.draw.line(screen, (255, 255, 0), sp, ep, max(1, int(3 * zoom)))
            
        if setting_start_dir and temp_start_pos:
            sp = world_to_screen(temp_start_pos)
            pygame.draw.circle(screen, (200, 0, 0), (int(sp[0]), int(sp[1])), max(2, int(5 * zoom)))
            pygame.draw.line(screen, (200, 200, 0), sp, mouse_pos, max(1, int(2 * zoom)))
            
        # Instructions
        instr = [
            "L-Click: Draw Checkpoints",
            "R-Click: Set Start Pos/Angle",
            "Mid-Click/Drag: Pan Map",
            "Scroll: Zoom In/Out",
            "Z: Undo last checkpoint",
            "S: Save to JSON",
            "C: Clear all"
        ]
        for i, text in enumerate(instr):
            t = font.render(text, True, (255, 255, 255), (0, 0, 0))
            screen.blit(t, (10, 10 + i * 25))
            
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left click
                    if setting_start_dir:
                        setting_start_dir = False
                        temp_start_pos = None
                    else:
                        if not current_line_start:
                            current_line_start = world_mouse
                        else:
                            checkpoints.append((current_line_start[0], current_line_start[1], world_mouse[0], world_mouse[1]))
                            current_line_start = None
                            
                elif event.button == 3: # Right click
                    if current_line_start:
                        current_line_start = None
                        
                    if not setting_start_dir:
                        setting_start_dir = True
                        temp_start_pos = world_mouse
                    else:
                        dx = world_mouse[0] - temp_start_pos[0]
                        dy = temp_start_pos[1] - world_mouse[1]
                        angle = math.degrees(math.atan2(dy, dx))
                        if angle < 0: angle += 360
                        
                        start_pos = {"x": temp_start_pos[0], "y": temp_start_pos[1], "angle": angle}
                        setting_start_dir = False
                        temp_start_pos = None
                        
                elif event.button == 2: # Middle click
                    panning = True
                    pan_start_mouse = mouse_pos
                    pan_start_cam = (cam_x, cam_y)
                    
            elif event.type == pygame.MOUSEWHEEL:
                old_world_x, old_world_y = screen_to_world(mouse_pos)
                if event.y > 0:
                    zoom = min(5.0, zoom * 1.1)
                elif event.y < 0:
                    zoom = max(0.1, zoom / 1.1)
                update_scaled_image()
                cam_x = mouse_pos[0] - old_world_x * zoom
                cam_y = mouse_pos[1] - old_world_y * zoom
                    
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 2:
                    panning = False
                    
            elif event.type == pygame.MOUSEMOTION:
                if panning:
                    cam_x = pan_start_cam[0] + (mouse_pos[0] - pan_start_mouse[0])
                    cam_y = pan_start_cam[1] + (mouse_pos[1] - pan_start_mouse[1])
                    
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_z:
                    if checkpoints:
                        checkpoints.pop()
                elif event.key == pygame.K_c:
                    checkpoints = []
                    start_pos = None
                elif event.key == pygame.K_s:
                    if not start_pos:
                        print("Please set a start position before saving.")
                        continue
                    
                    data = {
                        "track_name": track_name,
                        "start_position": start_pos,
                        "checkpoints": [
                            {"x1": cp[0], "y1": cp[1], "x2": cp[2], "y2": cp[3]} for cp in checkpoints
                        ]
                    }
                    out_path = f"{track_name}.json"
                    with open(out_path, 'w') as f:
                        json.dump(data, f, indent=2)
                    print(f"Saved to {out_path}")

        clock.tick(60)
        
    pygame.quit()

if __name__ == "__main__":
    main()
