import asyncio
import sys

import pygame
import torch
from pygame.locals import K_ESCAPE, K_SPACE, K_UP, KEYDOWN, QUIT

from .entities import (
    Background,
    Floor,
    GameOver,
    Pipes,
    Player,
    PlayerMode,
    Score,
    WelcomeMessage,
)
from .utils import GameConfig, Images, Sounds, Window
from ..DQN import Agent


class Flappy:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Flappy Bird")
        window = Window(288, 512)
        screen = pygame.display.set_mode((window.width, window.height))
        images = Images()
        self.num_episodes = 50
        self.config = GameConfig(
            screen=screen,
            clock=pygame.time.Clock(),
            fps=30,
            window=window,
            images=images,
            sounds=Sounds(),
        )
        self.agent = Agent(
            gamma=0.99,
            epsilon=1.0,
            lr=0.001,
            input_dims=[5],  # 假设状态向量长度为 5
            batch_size=32,
            n_actions=2,
            max_mem_size=100000,
            eps_end=0.01,
            eps_dec=5e-4
        )

    async def start(self):
        while True:
            self.background = Background(self.config)
            self.floor = Floor(self.config)
            self.player = Player(self.config)
            self.welcome_message = WelcomeMessage(self.config)
            self.game_over_message = GameOver(self.config)
            self.pipes = Pipes(self.config)
            self.score = Score(self.config)
            # await self.splash()
            await self.play()
            # await self.train()

            # await self.game_over()

    async def train(self):
        for episode in range(self.num_episodes):
            # 在每个回合开始时重置游戏
            self.reset_game()
            done = False
            score = 0
            state = self.closest_entity()

            while not done:
                # 选择并执行一个动作
                action = self.select_action()
                if action:  # 如果选择跳跃
                    self.player.flap()

                # 获取下一个状态、奖励和是否结束
                crossed = self.player.crossed()  # 检查是否越过水管
                done = self.player.collided()  # 检查是否碰撞

                # 计算下一个状态
                next_state = self.closest_entity()

                # 计算奖励
                currentReward = 10 * crossed - 100 * done
                # 如果小鸟在即将到来的水管的中间y轴位置，给予一些奖励
                # ... [奖励计算逻辑]

                # 存储转换
                # ... [存储逻辑]

                # 移动到下一个状态
                state = next_state

                # 执行优化步骤
                self.optimize_model()

                # 更新分数
                score += crossed

                if done:
                    break
    def reset_game(self):
        # 重置游戏状态的代码
        self.background = Background(self.config)
        self.floor = Floor(self.config)
        self.player = Player(self.config)
        self.pipes = Pipes(self.config)
        self.score = Score(self.config)
        self.player.set_mode(PlayerMode.NORMAL)

    async def splash(self):
        """Shows welcome splash screen animation of flappy bird"""

        self.player.set_mode(PlayerMode.SHM)

        while True:
            for event in pygame.event.get():
                self.check_quit_event(event)
                if self.is_tap_event(event):
                    return

            self.background.tick()
            self.floor.tick()
            self.player.tick()
            self.welcome_message.tick()

            pygame.display.update()
            await asyncio.sleep(0)
            self.config.tick()

    def check_quit_event(self, event):
        if event.type == QUIT or (
                event.type == KEYDOWN and event.key == K_ESCAPE
        ):
            pygame.quit()
            sys.exit()

    def is_tap_event(self, event):
        m_left, _, _ = pygame.mouse.get_pressed()
        space_or_up = event.type == KEYDOWN and (
                event.key == K_SPACE or event.key == K_UP
        )
        screen_tap = event.type == pygame.FINGERDOWN
        return m_left or space_or_up or screen_tap

    async def play(self):
        self.score.reset()
        self.player.set_mode(PlayerMode.NORMAL)

        while True:
            if self.player.collided(self.pipes, self.floor):
                return

            for i, pipe in enumerate(self.pipes.upper):
                if self.player.crossed(pipe):
                    self.score.add()

            for event in pygame.event.get():
                self.check_quit_event(event)
                if self.is_tap_event(event):
                    self.player.flap()

            self.background.tick()
            self.floor.tick()
            self.pipes.tick()
            self.score.tick()
            self.player.tick()

            pygame.display.update()
            await asyncio.sleep(0)
            self.config.tick()

    async def game_over(self):
        """crashes the player down and shows gameover image"""

        self.player.set_mode(PlayerMode.CRASH)
        self.pipes.stop()
        self.floor.stop()

        while True:
            for event in pygame.event.get():
                self.check_quit_event(event)
                if self.is_tap_event(event):
                    if self.player.y + self.player.h >= self.floor.y - 1:
                        return

            self.background.tick()
            self.floor.tick()
            self.pipes.tick()
            self.score.tick()
            self.player.tick()
            self.game_over_message.tick()

            self.config.tick()
            pygame.display.update()
            await asyncio.sleep(0)

    def closest_entity(self):
        # Assuming screen width as screen_width
        screen_height = self.config.window.height
        player = self.player

        # Set default values
        default_top = screen_height
        default_bot = 0

        # Get the position of the nearest upper and lower pipes
        nearest_pipe_x = 0
        nearest_entity_y_upper_bottom = default_bot
        nearest_entity_y_lower_top = default_top

        if len(self.pipes.upper) > 0 and len(self.pipes.lower) > 0:
            nearest_upper_pipe = self.pipes.upper[0]
            nearest_pipe_x = nearest_upper_pipe.x
            nearest_entity_y_upper_bottom = nearest_upper_pipe.rect.bottom
            nearest_lower_pipe = self.pipes.lower[0]
            nearest_entity_y_lower_top = nearest_lower_pipe.rect.top

        # Calculate the distance to the floor
        distance_to_floor = self.config.window.height - player.rect.y - player.rect.height / 2

        state = [
            player.rect.y - nearest_entity_y_upper_bottom,
            # Vertical distance from the player to the nearest upper pipe
            nearest_pipe_x - player.rect.x,  # Horizontal distance from the player to the nearest pipe
            nearest_entity_y_lower_top - player.rect.y,  # Vertical distance from the player to the nearest lower pipe
            distance_to_floor,  # Distance from the player to the floor
            player.vel_y  # Player's vertical velocity
        ]
        return state

    def select_action(self):
        state = self.closest_entity()
        state_tensor = torch.tensor([state], dtype=torch.float32).to(self.agent.Q_eval.device)
        action = self.agent.choose_action(state_tensor)
        return action