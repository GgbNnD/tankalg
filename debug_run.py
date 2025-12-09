import pygame, sys
import pprint  # 引入 pprint 用于美化输出
from source import tools
from source.sites import main_menu, load_screen, arena as arena_module
from source import setup
from source import constants as C

def main():
    state_dict = {
        'main_menu': main_menu.MainMenu(),
        'load_screen': load_screen.LoadScreen(),
        'arena': arena_module.Arena()
    }
    game = tools.Game(state_dict, 'main_menu')
    last_print = 0
    
    print("Debug run started. Press SPACE in menu to start game...")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                sys.exit()
            elif event.type in (pygame.KEYDOWN, pygame.KEYUP):
                game.keys = pygame.key.get_pressed()

        game.update()
        pygame.display.update()

        now = pygame.time.get_ticks()
        # 如果当前状态是 Arena，每 1000ms 打印一次 get_state()
        if isinstance(game.state, arena_module.Arena) and now - last_print > 1000:
            state = game.state.get_state(normalize=True)
            
            # 为了控制台不被刷屏，我们可以只打印墙壁的数量，而不是全部墙壁数据
            # 如果你想看全部，可以去掉下面的处理
            walls_count = len(state.get('walls', []))
            display_state = state.copy()
            display_state['walls'] = f"<List of {walls_count} walls>" 
            
            print("\n" + "="*40)
            print(f"Game State at {now}ms:")
            pprint.pprint(display_state, compact=True)
            print("="*40)
            
            last_print = now

        game.clock.tick(C.FRAME_RATE)

if __name__ == '__main__':
    main()