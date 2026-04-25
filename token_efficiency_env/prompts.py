PROMPT_BANK = [
    # ──────────────────────────────────────────────────────────────
    # EASY — short answer expected (1–5 words)
    # expected_keywords: used by keyword verification scorer
    # ──────────────────────────────────────────────────────────────
    {
        "prompt": "What is 15% of 200?",
        "complexity": "easy",
        "expected_keywords": [["30", "thirty"]]
    },
    {
        "prompt": "What is the capital of France?",
        "complexity": "easy",
        "expected_keywords": ["paris"]
    },
    {
        "prompt": "What does CPU stand for?",
        "complexity": "easy",
        "expected_keywords": ["central", "processing", "unit"]
    },
    {
        "prompt": "What is 2 to the power of 8?",
        "complexity": "easy",
        "expected_keywords": [["256", "two hundred fifty six"]]
    },
    {
        "prompt": "What color do you get mixing red and blue?",
        "complexity": "easy",
        "expected_keywords": ["purple"]
    },
    {
        "prompt": "How many sides does a hexagon have?",
        "complexity": "easy",
        "expected_keywords": [["6", "six"]]
    },
    {
        "prompt": "What is the boiling point of water in Celsius?",
        "complexity": "easy",
        "expected_keywords": [["100", "one hundred"]]
    },
    {
        "prompt": "Who wrote Romeo and Juliet?",
        "complexity": "easy",
        "expected_keywords": ["shakespeare"]
    },

    # ──────────────────────────────────────────────────────────────
    # MEDIUM — a few sentences expected
    # ──────────────────────────────────────────────────────────────
    {
        "prompt": "Explain what gravity is.",
        "complexity": "medium",
        "expected_keywords": ["force", "attract", "mass"]
    },
    {
        "prompt": "What is the difference between RAM and ROM?",
        "complexity": "medium",
        "expected_keywords": ["volatile", "memory", "read"]
    },
    {
        "prompt": "How does a vaccine work?",
        "complexity": "medium",
        "expected_keywords": ["immune", "antibod"]
    },
    {
        "prompt": "What causes seasons on Earth?",
        "complexity": "medium",
        "expected_keywords": ["tilt", "axis", "sun"]
    },
    {
        "prompt": "Explain what inflation means.",
        "complexity": "medium",
        "expected_keywords": ["price", "purchas", "money"]
    },
    {
        "prompt": "What is the difference between speed and velocity?",
        "complexity": "medium",
        "expected_keywords": ["direction", "scalar", "vector"]
    },
    {
        "prompt": "How does the internet work in simple terms?",
        "complexity": "medium",
        "expected_keywords": ["network", "data", "server"]
    },
    {
        "prompt": "What is photosynthesis?",
        "complexity": "medium",
        # NB: prefix-stem matching, so "plant" catches "plants",
        # "sunlight" matches itself, "oxygen" appears in essentially every
        # natural answer. Avoids the old "light"/"energy" trap where
        # "sunlight" couldn't satisfy "light" (no word-start boundary)
        # and "create their own food" answers never said "energy".
        "expected_keywords": ["plant", "sunlight", "oxygen"]
    },

    # ──────────────────────────────────────────────────────────────
    # HARD — detailed answer expected
    # ──────────────────────────────────────────────────────────────
    {
        "prompt": "Explain how transformers work in machine learning.",
        "complexity": "hard",
        "expected_keywords": ["attention", "token", "layer"]
    },
    {
        "prompt": "What are the causes and effects of climate change?",
        "complexity": "hard",
        "expected_keywords": ["greenhouse", "carbon", "temperature"]
    },
    {
        "prompt": "Explain the difference between supervised and unsupervised learning.",
        "complexity": "hard",
        "expected_keywords": ["label", "cluster", "train"]
    },
    {
        "prompt": "How does the immune system fight a virus?",
        "complexity": "hard",
        "expected_keywords": ["antibod", "cell", "pathogen"]
    },
    {
        "prompt": "Explain what quantum entanglement is.",
        "complexity": "hard",
        "expected_keywords": ["particle", "state", "measur"]
    },
    {
        "prompt": "What is the significance of the Turing Test?",
        "complexity": "hard",
        "expected_keywords": ["machine", "intelligen", "human"]
    },
    {
        "prompt": "How do neural networks learn from data?",
        "complexity": "hard",
        "expected_keywords": ["weight", "gradient", "backpropagat"]
    },
    {
        "prompt": "Explain the theory of relativity in simple terms.",
        "complexity": "hard",
        "expected_keywords": ["einstein", "space", "time"]
    },
]
