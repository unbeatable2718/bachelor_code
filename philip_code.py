class Time_Variance:

    @staticmethod
    def relative_track_length(cos_theta):
        """Calculate relative track length for cos_theta."""
        k_radius = 1.8
        k_height = 1.2
        sin_theta = np.sqrt(1 - pow(cos_theta, 2))
        return 1. / (cos_theta + 2 * k_height * sin_theta / (np.pi * k_radius))

    @staticmethod
    def mean_t50(cos_theta, distance):
        """Calculate the mean of t50."""
        # TODO: This was updated after implemented here, but I compare with
        #       data which used this version
        nanosec = Units.nanosecond  # pow(10, -9)
        return 0.8 * nanosec * (0.53 * cos_theta - 0.11) * distance

    @classmethod
    def limit_t50(cls, t50, cos_theta, distance):
        """Limit t50 to half of mean_t50."""
        limit = 0.5 * cls.mean_t50(cos_theta, distance)
        if t50 < limit:
            return limit
        else:
            return t50

    @staticmethod
    def form_2006(n, t50, a2, b2):
        """Return variance: C. Bonifazi, A. Letessier-Selvon: GAP2006-16."""
        return a2 * pow(2 * t50 / n, 2) * (n - 1) / (n + 1) + b2

    @staticmethod
    def form_2007(n, t50, tmin, a2, b2):
        """Return variance: M. Horvat, D. Veberic: GAP2007-057."""
        return a2 * pow((t50 + tmin) / (n + 1), 2) * n / (n + 2) + b2

    @staticmethod
    def form_2012(n, t50, a2, b2):
        """Return variance: GAP2012-145."""
        return a2 * pow(2 * t50 / (n - 1), 2) * n / (n + 2) + b2

    @classmethod
    def get_time_sigma2(cls, signal, t50, cos_theta, distance, model):
        n = signal / cls.relative_track_length(cos_theta)
        nanosec = Units.nanosecond  # pow(10, -9)
        nanosec2 = Units.nanosecond ** 2  # pow(10, -18)
        t50 *= nanosec

        if model == "eICRC2005":
            val = (600 * nanosec + 1.2 * pow(t50 / signal, 2))
            return val * (0.4 + 1.2 * cos_theta)
        elif model == "eGAP2006_016":
            return cls.form_2006(n, t50, 1, 147 * nanosec2)
        elif model == "eNIMA":
            return cls.form_2006(n, t50, 0.99, 147 * nanosec2)
        elif model == "eCDASv4r4":
            return cls.form_2007(n, t50, 15 * nanosec, 5, 134 * nanosec2)
        elif model == "eGAP2007_057":
            if n < 2:
                return cls.form_2007(2, t50, 10 * nanosec, 2.4, 134 * nanosec2)
            else:
                return cls.form_2007(n, t50, 10 * nanosec, 2.4, 134 * nanosec2)
        elif model == "eCDASv4r6":
            if n < 2:
                return cls.form_2006(2, t50, 0.36, 212 * nanosec2)
            else:
                return cls.form_2006(n, t50, 0.36, 212 * nanosec2)
        elif model == "eCDASv4r8":
            dt = cls.limit_t50(t50, cos_theta, distance)
            a2 = 0.67612 + cos_theta * (0.16106 - 0.47641 * cos_theta)
            b2 = (128 + cos_theta * (413 * cos_theta - 184)) * nanosec2
            if n < 2:
                return cls.form_2006(2, dt, a2, b2)
            else:
                return cls.form_2006(n, dt, a2, b2)
        elif model == "eCDASv5r0":
            # GAP2018-048
            dt = cls.limit_t50(t50, cos_theta, distance)
            a2 = 0.64871 + cos_theta * (0.22365 - 0.49971 * cos_theta)
            b2 = (141.24 + cos_theta * (412.19 * cos_theta - 208.9)) * nanosec2
            if n < 2:
                return cls.form_2006(2, dt, a2, b2)
            else:
                return cls.form_2006(n, dt, a2, b2)
        elif model == "eGAP2012_145":
            mean_t50 = cls.mean_t50(cos_theta, distance)
            if n < 4 and t50 < mean_t50:
                dt = mean_t50
            else:
                dt = t50
            # a and b are the same as in eCDASv5r0
            a2 = 0.64871 + cos_theta * (0.22365 - 0.49971 * cos_theta)
            b2 = (141.24 + cos_theta * (412.19 * cos_theta - 208.9)) * nanosec2
            if n < 2:
                return cls.form_2012(2, dt, a2, b2)
            else:
                return cls.form_2012(n, dt, a2, b2)
        elif model =="eGAP2018_045":
            """Alan Coleman's time variance."""
            gap2018_045_params = [-0.88, 0.31, -2.27, -2.29, 1.94, 4.31]
            a, b, c, d, e, f = gap2018_045_params
            x = a * np.log(distance) + b * np.log(signal)
            return np.exp(c / (1 + np.exp(- (x - d) / e)) + f)
        else:
            raise Exception("Specify a known time variance model!")