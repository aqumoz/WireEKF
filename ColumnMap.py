import numpy as np

class Data:
    """A class for structureing data in the way that WireEKF expects"""
    def __init__(self):
        self.time : float
        self.ft : tuple[float, float, float] | None
        self.torque : tuple[float, float, float] | None
        self.p : dict[int, tuple[float, float, float]] = {}
        self.q : dict[int, tuple[float, float, float, float]] = {}
        self.v : dict[int, tuple[float, float, float]] = {} # Added for velocities

    def to_x(self) -> np.ndarray:
        """Convert the position data into a state vector format (3*n, 1)"""
        n = len(self.p)
        x_vec = np.zeros((3 * n, 1))
        for i in range(1, n + 1):
            idx = (i - 1) * 3
            x_vec[idx : idx + 3, 0] = self.p[i]
        return x_vec
    
    def __str__(self):
        return f"Data(time={self.time}, ft={self.ft}, torque={self.torque}, p={self.p}, q={self.q}, v={self.v})"


class ColumnMap:
    """Restructures the data to a known format for WireEKF"""
    def __init__(self, map: str | None = None):
        default_map = "time,p1_x,p1_y,p1_z,q1_w,q1_x,q1_y,q1_z,p2_x,p2_y,p2_z,q2_w,q2_x,q2_y,q2_z,p3_x,p3_y,p3_z,q3_w,q3_x,q3_y,q3_z,p4_x,p4_y,p4_z,q4_w,q4_x,q4_y,q4_z,p5_x,p5_y,p5_z,q5_w,q5_x,q5_y,q5_z,p6_x,p6_y,p6_z,q6_w,q6_x,q6_y,q6_z,p7_x,p7_y,p7_z,q7_w,q7_x,q7_y,q7_z,p8_x,p8_y,p8_z,q8_w,q8_x,q8_y,q8_z,p9_x,p9_y,p9_z,q9_w,q9_x,q9_y,q9_z,p10_x,p10_y,p10_z,q10_w,q10_x,q10_y,q10_z,ft_x,ft_y,ft_z,torque_x,torque_y,torque_z"
        map = map if map else default_map
        self.map = [x.strip() for x in map.split(",")]

        self.col_time : int | None = None
        self.col_ft : list[int] | None = None # [col_ft_x, col_ft_y, col_ft_z]
        self.col_torque : list[int] | None = None # [col_torque_x, col_torque_y, col_torque_z]
        self.col_p : dict[int, list[int]] = {} # {1: [p1_x, p1_y, p1_z], 2: [p2_x, p2_y, p2_z], ...}
        self.col_q : dict[int, list[int]] = {} # {1: [q1_w, q1_x, q1_y, q1_z], 2: [q2_w, q2_x, q2_y, q2_z], ...}
        
        i = 0
        for col in self.map:
            if col == "time":
                self.col_time = i

            elif col.startswith("ft_"):
                if self.col_ft is None:
                    self.col_ft = [0, 0, 0]
                self.col_ft[{'x': 0, 'y': 1, 'z': 2}[col[-1]]] = i
            
            elif col.startswith("torque_"):
                if self.col_torque is None:
                    self.col_torque = [0, 0, 0]
                self.col_torque[{'x': 0, 'y': 1, 'z': 2}[col[-1]]] = i

            elif col.startswith("p"):
                number = int(col[1:col.find("_")])
                if len(self.col_p) < number:
                    self.col_p[number - 1] = [0,0,0]
                self.col_p[number - 1][{'x': 0, 'y': 1, 'z': 2}[col[-1]]] = i
            
            elif col.startswith("q"):
                number = int(col[1:col.find("_")])
                if len(self.col_q) < number:
                    self.col_q[number - 1] = [0,0,0,0]
                self.col_q[number - 1][{'w': 0, 'x': 1, 'y': 2, 'z': 3}[col[-1]]] = i
            
            i += 1
        
        self.n_nodes : int = len(self.col_p)

    def from_list(self, data_line : list[float]) -> Data:
        data = Data()
        if self.col_time is None:
            raise ValueError("Column map must include a 'time' column.")
        data.time = data_line[self.col_time]
        data.ft = (data_line[self.col_ft[0]], data_line[self.col_ft[1]], data_line[self.col_ft[2]]) if self.col_ft is not None else None
        data.torque = (data_line[self.col_torque[0]], data_line[self.col_torque[1]], data_line[self.col_torque[2]]) if self.col_torque is not None else None
        data.p = {i + 1: (data_line[col[0]], data_line[col[1]], data_line[col[2]]) for i, col in self.col_p.items()}
        data.q = {i + 1: (data_line[col[0]], data_line[col[1]], data_line[col[2]], data_line[col[3]]) for i, col in self.col_q.items()}
        return data

    def from_csv_line(self, data_line : str) -> Data:
        return self.from_list([float(v) for v in data_line.split(",")])



if __name__ == "__main__":
    # Example usage
    column_map = ColumnMap("time,p1_x,p1_y,p1_z,q1_w,q1_x,q1_y,q1_z,p3_x,p3_y,p3_z,q3_w,q3_x,q3_y,q3_z,ft_x,ft_y,ft_z,torque_x,torque_y,torque_z")
    print("Column map:", column_map.map)
    print("Time column index:", column_map.col_time)
    print("Force-torque columns indices:", column_map.col_ft)
    print("Torque columns indices:", column_map.col_torque)
    print("Position columns indices:", column_map.col_p)
    print("Orientation columns indices:", column_map.col_q)

    # Example data line (replace with actual data)
    data_line = "0,0,0,0,0,0.707106781186547,0,0.707106781186548,0.100000018670281,-4.91390105816864E-07,-3.16055503654019E-06,4.56672794208245E-07,0.713252042646815,-4.6473378087563E-07,0.700907642745977,18.0373678792504,-2.7683073716428,0.234135938565286,-4.06780931318941E-06,-8.09816623404048,0.000604460752677451"
    print("Example data line:", data_line)
    data = column_map.from_csv_line(data_line)
    print("Parsed data:", data.__dict__)