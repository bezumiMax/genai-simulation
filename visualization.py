from generator_learning import GeneratorTrainer
from discriminator_learning import DiscriminatorTrainer
import numpy
import matplotlib.pyplot as plt
from tkinter import Tk

class TrainingVisualizer:
    def __init__(self):
        self.losses_g = []
        self.losses_d = []
        self.success_rates = []
        plt.ion()  # Интерактивный режим
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 4))

    def update_plots(self, level, gen_loss, dis_loss, success_rate):
        self.losses_g.append(gen_loss)
        self.losses_d.append(dis_loss)
        self.success_rates.append(success_rate)

        self.ax1.clear()
        self.ax2.clear()

        # График потерь
        self.ax1.plot(self.losses_g, 'b-', label='Generator Loss', linewidth=2)
        self.ax1.plot(self.losses_d, 'r-', label='Discriminator Loss', linewidth=2)
        self.ax1.set_title('🎯 Losses')
        self.ax1.set_xlabel('Level')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)

        # График успеваемости
        self.ax2.plot(self.success_rates, 'g-', label='Success Rate', linewidth=2, marker='o')
        self.ax2.set_title('📈 Success Rate')
        self.ax2.set_xlabel('Level')
        self.ax2.set_ylim(0, 1)
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

    def print_status(self, level, gen_loss, dis_loss, success_rate):
        """Выводит статус в консоль"""
        print(f"\n🎯 Level {level}")
        print(f"   Generator Loss: {gen_loss:.4f}")
        print(f"   Discriminator Loss: {dis_loss:.4f}")
        print(f"   Success Rate: {success_rate:.2%}")
        print("-" * 40)
