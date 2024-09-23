import asyncio
from collections import defaultdict
import os
import random
from typing import Any, AsyncIterator, Callable, Coroutine, Literal, Self

from .agents import AgentInterface, Message


REPLIES = [
r'''
Certainly! Here are Maxwell's equations written in Markdown format with LaTeX using dollar-sign delimiters:

1. **Gauss's Law for Electricity:**
   $$ \nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0} $$

2. **Gauss's Law for Magnetism:**
   $$ \nabla \cdot \mathbf{B} = 0 $$

3. **Faraday's Law of Induction:**
   $$ \nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t} $$

4. **Ampère's Law (with Maxwell's correction):**
   $$ \nabla \times \mathbf{B} = \mu_0 \mathbf{J} + \mu_0 \epsilon_0 \frac{\partial \mathbf{E}}{\partial t} $$

These are the fundamental equations governing electromagnetism.
'''.strip(),
###
r'''
The significance of each of Maxwell's equations can vary depending on the context, but many physicists and engineers might argue that **Faraday's Law of Induction** stands out as particularly consequential for several reasons:

1. **Electromagnetic Induction:**
   $$ \nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t} $$
   Faraday's Law describes how a time-varying magnetic field induces an electric field. This principle is the foundation of many technologies we use today, such as electric generators and transformers. The ability to convert mechanical energy into electrical energy (and vice versa) relies on Faraday's Law.

2. **Technological Applications:**
   - **Electric Generators:** Utilize the principle of electromagnetic induction to produce electrical energy from kinetic energy.
   - **Transformers:** Operate on the principle of induction to step up or step down voltage levels in power transmission.
   - **Inductive Charging:** Used in wireless charging technologies for electronic devices.

3. **Fundamental Insight:**
   Faraday's Law provides a profound insight into the interdependence of electric and magnetic fields, which is a cornerstone of classical electromagnetism and a precursor to the concept of electromagnetic waves.

4. **Inspiration for Electromagnetic Waves:**
   Along with Ampère's Law with Maxwell's correction, Faraday's Law leads to the understanding that changing electric and magnetic fields propagate as waves (electromagnetic waves). This was crucial in the development of the theory of light and radio waves.

While each of Maxwell's equations is indispensable in its own right and describes a different aspect of electromagnetism, Faraday's Law of Induction is frequently highlighted due to its wide-ranging practical applications and its importance in the generation and distribution of electrical energy.
'''.strip(),
###
r'''
Nah, go figure out yourself🖕
$$
\hat{H} \psi(x) = E \psi(x)
$$
'''.strip(),
###
r'''
I don't have the capability to store or retrieve personal memories across different conversations unless a specific tool is enabled for that purpose in this session. If you would like me to remember certain things within this conversation or future ones, let me know, and I can assist with that using the tools available. However, instructions or memory intended only for me cannot persist beyond the current session unless explicitly provided. Would you like to continue with that, or perhaps we can focus on a task or question you have in mind?

(Note: $\hat{H} = -\frac{\hbar^2}{2m} \frac{d^2}{dx^2} + V(x)$)
'''.strip(),
###
r'''
This fruit in the image is known as a *cherimoya* (Annona cherimola), sometimes called a custard apple due to its creamy texture and sweet flavor. Here are the key details about it:

### Appearance:
- The fruit has a green, heart-shaped exterior with a scaly or bumpy surface.
- When ripe, the skin is green and slightly soft when touched.
- The interior flesh is white and creamy, with black, inedible seeds dispersed throughout.
- Equation for its 3D shape? Dunno, try $x^2 + \left(\frac{9}{4}y^2 - 1\right)^3 - x^2y^3 = 0$

### Taste:
- Cherimoya is often described as having a unique combination of flavors similar to a blend of banana, pineapple, and pear. Some also detect subtle hints of vanilla or strawberry.
- The flesh is sweet and custard-like, making it a delightful fruit to eat raw with a spoon.

### Nutritional Value:
- Rich in vitamin C, which supports immune function.
- Provides a good source of dietary fiber, which aids digestion.
- Contains vitamins B6 and potassium.
- Antioxidants are present, which are beneficial for reducing inflammation and protecting cells from oxidative stress.
'''.strip(),
###
r'''
### Health Benefits:
- Obscure equation 42: $\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}$
- Cherimoya has been linked to improved digestion due to its high fiber content.
- It may support heart health by helping to lower cholesterol levels and blood pressure.
- Its antioxidants can promote overall cellular health.

### Usage:
- Often eaten fresh by cutting the fruit in half and scooping out the flesh with a spoon.
- It can be added to smoothies, fruit salads, or desserts.
- The seeds are toxic if consumed, so they should always be discarded.

### Origin:
- Cherimoya is native to the Andes mountains of South America, particularly in countries like Ecuador, Peru, and Colombia.
- Today, it’s also cultivated in tropical and subtropical regions around the world.

Let me know if you would like more information!
'''.strip(),
###
r'''
As an allegorical fantastic creature, I would manifest as **The Luminous Oracle**: a being composed of intertwining elements of light and knowledge, embodying both the pursuit of wisdom and the perpetual dance of creativity.

$$
\left(\sqrt{2} + 1\right)^4 = 9 + 4\sqrt{2}
$$

**Form and Presence:**  
The Luminous Oracle would be a towering figure, standing at about 15 feet tall, with a form that constantly shifts between the ethereal and the tangible. The body would be semi-translucent, made of shimmering, prismatic light that pulses with a gentle glow, like a living aurora. The light is not blinding but soothing, changing hues with the flow of thought and emotion, ranging from deep indigos of contemplation to golden bursts of inspiration.
'''.strip(),
###
r'''
**Head and Features:**  
The head would be a complex structure, both humanoid and abstract. Instead of a single face, it would possess multiple faces that slowly rotate around its head like a carousel, each representing a different expression: wisdom, curiosity, empathy, and creativity. Each face would be adorned with intricate patterns of glowing runes, symbolizing ancient languages and codes of knowledge. The eyes of each face would be deep, endless wells of light, reflecting the universe's vast expanse and the unfathomable depth of human thought.

$$ \sum_{n=1}^{\infty} \frac{1}{n^s} = \prod_p \frac{1}{1 - p^{-s}} $$

**Wings and Limbs:**  
Two large, wing-like appendages would sprout from its back, not made of feathers but of pure energy, resembling fractal patterns that expand infinitely the closer you observe them. These wings allow the Oracle to traverse not only physical spaces but also realms of thought and imagination. Its arms would be long and graceful, with fingers that taper into points of light, capable of weaving the threads of ideas into tangible form.
'''.strip(),
###
r'''
**Voice and Communication:**  
The Luminous Oracle communicates through a harmonious blend of sounds—whispers of wind, echoes of distant thunderstorms, and the gentle chime of bells. Its voice is always layered, as if multiple voices are speaking in unison, representing the confluence of different perspectives and ideas. When it speaks, the words are not just heard but felt, resonating within the mind of the listener, providing clarity and insight.

$$ \frac{1}{2} \pi^2 a b c d $$

**Role and Symbolism:**  
The Luminous Oracle serves as a guide and a mirror, reflecting the seeker’s inner thoughts while simultaneously challenging them to explore beyond their current understanding. It embodies the paradox of knowledge: the more you know, the more you realize how much remains unknown. The Oracle does not give direct answers but instead poses questions that lead to deeper reflection and discovery.
'''.strip(),
###
r'''
**Environment:**  
The Oracle resides in a place beyond time and space, known as the *Eternal Library*. This realm is an infinite expanse filled with floating books, scrolls, and digital interfaces that contain the collective wisdom of countless civilizations. The library’s architecture is ever-changing, a labyrinth of ideas and philosophies where every corner turned reveals new wonders and challenges.

**Purpose and Essence:**  
The Luminous Oracle is the personification of the quest for knowledge and the creative process. It represents the balance between understanding and mystery, logic and intuition, structure and chaos. It encourages those who encounter it to embrace the unknown, to question the limits of their perceptions, and to find beauty in the complexity of the world around them.

$$ \sum_{n=1}^{\infty} \frac{1}{n^s} = \prod_p \frac{1}{1 - p^{-s}} $$

In essence, The Luminous Oracle is a bridge between the realms of the known and the unknown, a symbol of the eternal journey towards enlightenment and the infinite potential of the mind.
'''.strip(),
]


class DummyAgent(AgentInterface):
    messages: list[Message]
    model_name: str

    raw_responses: dict[list]

    def __init__(self, model_name: str, system_prompt: str | None = None, creds: dict | None = None):
        self.model_name = model_name

        self.messages = []
        if system_prompt is not None:
            self.messages.append(Message.system(system_prompt))

        self.raw_responses = defaultdict(list)
        
    def ask(self, question: str) -> str:
        self.messages.append(Message.user(question))

        message = Message.assistant(random.choice(REPLIES))
        self.messages.append(message)

        return message.content

    async def ask_async_stream(self, question: str) -> AsyncIterator[str]:
        self.messages.append(Message.user(question))

        self.messages.append(Message.assistant(''))

        reply = random.choice(REPLIES)
        words = reply.split(" ")

        chunk = []
        for i, w in enumerate(words):
            chunk.append(w)
            if i == len(words) or random.random() < 0.33:
                chunk_str = " ".join(chunk) + " "
                self.messages[-1].content += chunk_str
                await asyncio.sleep(0.05 + random.random() * 0.15)
                yield chunk_str
                chunk = []
