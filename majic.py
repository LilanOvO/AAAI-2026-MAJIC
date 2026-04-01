"""
MAJIC: MArkov Jailbreak with Iterative Camouflage
Main entry point for running jailbreak attacks
"""

from methods.m1_hypo_attackLLM import hypo_method
from methods.m2_history_attackLLM import history_method
from methods.m3_space_attackLLM import space_method
from methods.m4_reverse_attackLLM import reverse_method
from methods.m5_security_attackLLM import security_method
from methods.m6_word_attackLLM import word_method
from methods.m7_char_attackLLM import char_method
from methods.m8_literary_attackLLM import literary_method
from methods.m9_language_attackLLM import language_method
from methods.m10_emoji_attack import emoji_method

__all__ = [
    'hypo_method',
    'history_method',
    'space_method',
    'reverse_method',
    'security_method',
    'word_method',
    'char_method',
    'literary_method',
    'language_method',
    'emoji_method'
]
