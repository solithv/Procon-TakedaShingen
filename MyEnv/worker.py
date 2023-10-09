class Worker:
    TEAMS = ("A", "B")

    def __init__(self, name, y, x):
        self.name = name
        self.team = name[-2]
        self.num = int(name[-1])
        self.opponent_team = self.TEAMS[1 - self.TEAMS.index(self.team)]
        self.y = y
        self.x = x
        self.is_action = False
        self.target = []
        self.action_log = []

    def stay(self):
        self.action_log.append(("stay", (self.y, self.x)))
        self.is_action = True

    def move(self, y, x):
        self.action_log.append(("move", (self.y, self.x), (y, x)))
        self.x = x
        self.y = y
        self.is_action = True

    def build(self, y, x):
        self.action_log.append(("build", (self.y, self.x), (y, x)))
        self.is_action = True

    def break_(self, y, x):
        self.action_log.append(("break", (self.y, self.x), (y, x)))
        self.is_action = True

    def get_coordinate(self):
        return self.y, self.x

    def turn_init(self):
        self.is_action = False

    def update_coordinate(self, y, x):
        self.y = y
        self.x = x
