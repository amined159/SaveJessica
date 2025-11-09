"""
Data collection and analysis functions for the Morty Express Challenge.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from api_client import SphinxAPIClient


class DataCollector:
    """Collects and analyzes data from the challenge."""
    
    def __init__(self, client: SphinxAPIClient):
        """
        Initialize the data collector.
        
        Args:
            client: SphinxAPIClient instance
        """
        self.client = client
        self.trips_data = []
    
    def explore_planet(self, planet: int, num_trips: int, morty_count: int = 1) -> pd.DataFrame:
        """
        Send multiple trips to a single planet to observe its behavior.
        
        Args:
            planet: Planet index (0, 1, or 2)
            num_trips: Number of trips to make
            morty_count: Number of Morties per trip (1-3)
            
        Returns:
            DataFrame with trip data
        """
        print(f"\nExploring {self.client.get_planet_name(planet)}...")
        print(f"Sending {num_trips} trips with {morty_count} Morties each")
        
        trips = []
        
        for i in range(num_trips):
            try:
                result = self.client.send_morties(planet, morty_count)
                
                trip_data = {
                    'trip_number': i + 1,
                    'planet': planet,
                    'planet_name': self.client.get_planet_name(planet),
                    'morties_sent': result['morties_sent'],
                    'survived': result['survived'],
                    'steps_taken': result['steps_taken'],
                    'morties_in_citadel': result['morties_in_citadel'],
                    'morties_on_planet_jessica': result['morties_on_planet_jessica'],
                    'morties_lost': result['morties_lost']
                }
                
                trips.append(trip_data)
                self.trips_data.append(trip_data)
                
                if (i + 1) % 10 == 0:
                    print(f"  Completed {i + 1}/{num_trips} trips")
                
            except Exception as e:
                print(f"Error on trip {i + 1}: {e}")
                break
        
        df = pd.DataFrame(trips)
        
        if len(df) > 0:
            survival_rate = df['survived'].mean() * 100
            print(f"\nResults for {self.client.get_planet_name(planet)}:")
            print(f"  Survival Rate: {survival_rate:.2f}%")
            print(f"  Trips Completed: {len(df)}")
            print(f"  Morties Saved: {df['survived'].sum() * morty_count}")
            print(f"  Morties Lost: {df['morties_lost'].iloc[-1]}")
        
        return df
    
    def explore_all_planets(self, trips_per_planet: int = 30, morty_count: int = 1) -> pd.DataFrame:
        """
        Explore all three planets to compare their behaviors.
        
        Args:
            trips_per_planet: Number of trips to make to each planet
            morty_count: Number of Morties per trip
            
        Returns:
            Combined DataFrame with all trip data
        """
        print("\n" + "="*60)
        print("EXPLORING ALL PLANETS")
        print("="*60)
        
        # Start a new episode
        self.client.start_episode()
        self.trips_data = []
        
        all_data = []
        
        for planet in [0, 1, 2]:
            df = self.explore_planet(planet, trips_per_planet, morty_count)
            all_data.append(df)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        print("\n" + "="*60)
        print("SUMMARY ACROSS ALL PLANETS")
        print("="*60)
        
        summary = combined_df.groupby('planet_name').agg({
            'survived': ['sum', 'mean', 'count']
        })
        
        for planet_name in combined_df['planet_name'].unique():
            planet_data = combined_df[combined_df['planet_name'] == planet_name]
            survival_rate = planet_data['survived'].mean() * 100
            total_saved = planet_data['survived'].sum() * morty_count
            
            print(f"\n{planet_name}:")
            print(f"  Survival Rate: {survival_rate:.2f}%")
            print(f"  Morties Saved: {total_saved}")
            print(f"  Trips: {len(planet_data)}")
        
        return combined_df
    
    def calculate_moving_average(self, df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """
        Calculate moving average of survival rates for each planet.
        
        Args:
            df: DataFrame with trip data
            window: Window size for moving average
            
        Returns:
            DataFrame with moving averages
        """
        result = df.copy()
        
        for planet in df['planet'].unique():
            mask = result['planet'] == planet
            planet_survived = result.loc[mask, 'survived'].astype(int)
            result.loc[mask, 'survival_ma'] = planet_survived.rolling(
                window=window, min_periods=1
            ).mean()
        
        return result
    
    def analyze_risk_changes(self, df: pd.DataFrame) -> Dict:
        """
        Analyze how risk changes over time for each planet.
        
        Args:
            df: DataFrame with trip data
            
        Returns:
            Dictionary with risk analysis for each planet
        """
        analysis = {}
        
        for planet in df['planet'].unique():
            planet_data = df[df['planet'] == planet].copy()
            planet_name = planet_data['planet_name'].iloc[0]
            
            # Split into early and late trips
            mid_point = len(planet_data) // 2
            early = planet_data.iloc[:mid_point]
            late = planet_data.iloc[mid_point:]
            
            early_survival = early['survived'].mean()
            late_survival = late['survived'].mean()
            change = late_survival - early_survival
            
            analysis[planet_name] = {
                'planet': planet,
                'early_survival_rate': early_survival * 100,
                'late_survival_rate': late_survival * 100,
                'change': change * 100,
                'trend': 'improving' if change > 0 else 'worsening',
                'total_trips': len(planet_data),
                'overall_survival_rate': planet_data['survived'].mean() * 100
            }
        
        return analysis
    
    def get_best_planet(self, df: pd.DataFrame, consider_trend: bool = True) -> Tuple[int, str]:
        """
        Determine the best planet to use based on collected data.
        
        Args:
            df: DataFrame with trip data
            consider_trend: Whether to consider trend in addition to survival rate
            
        Returns:
            Tuple of (planet_index, planet_name)
        """
        if consider_trend:
            analysis = self.analyze_risk_changes(df)
            
            # Score based on late survival rate (more recent data)
            best_planet = max(
                analysis.items(),
                key=lambda x: x[1]['late_survival_rate']
            )
            
            planet_name = best_planet[0]
            planet_index = int(best_planet[1]['planet'])  # Convert to Python int
            
        else:
            # Simple overall survival rate
            planet_rates = df.groupby(['planet', 'planet_name'])['survived'].mean()
            best = planet_rates.idxmax()
            planet_index, planet_name = best
            planet_index = int(planet_index)  # Convert to Python int
        
        return planet_index, planet_name
    
    def save_data(self, filename: str = "trips_data.csv"):
        """
        Save collected trip data to a CSV file.
        
        Args:
            filename: Name of the CSV file
        """
        if self.trips_data:
            df = pd.DataFrame(self.trips_data)
            df.to_csv(filename, index=False)
            print(f"\nData saved to {filename}")
        else:
            print("No data to save")
    
    def load_data(self, filename: str = "trips_data.csv") -> pd.DataFrame:
        """
        Load trip data from a CSV file.
        
        Args:
            filename: Name of the CSV file
            
        Returns:
            DataFrame with trip data
        """
        df = pd.read_csv(filename)
        self.trips_data = df.to_dict('records')
        print(f"\nLoaded {len(df)} trips from {filename}")
        return df
    
    def alternate_planets(self, total_trips: int, morty_count: int = 1) -> pd.DataFrame:
        """
        Send Morties in a cyclic order through all planets (0 → 1 → 2 → 0 → ...).

        Args:
            total_trips: Total number of trips to make across all planets
            morty_count: Number of Morties per trip

        Returns:
            DataFrame with trip data
        """
        print("\n" + "="*60)
        print("ALTERNATING BETWEEN PLANETS")
        print("="*60)

        # Start a new episode before sending Morties
        self.client.start_episode()
        self.trips_data = []

        trips = []
        planets = [0, 1, 2]

        for i in range(total_trips):
            planet = planets[i % len(planets)]  # cycle through 0, 1, 2
            planet_name = self.client.get_planet_name(planet)

            try:
                result = self.client.send_morties(planet, morty_count)

                trip_data = {
                    'trip_number': i + 1,
                    'planet': planet,
                    'planet_name': planet_name,
                    'morties_sent': result['morties_sent'],
                    'survived': result['survived'],
                    'steps_taken': result['steps_taken'],
                    'morties_in_citadel': result['morties_in_citadel'],
                    'morties_on_planet_jessica': result['morties_on_planet_jessica'],
                    'morties_lost': result['morties_lost']
                }

                trips.append(trip_data)
                self.trips_data.append(trip_data)

                if (i + 1) % 10 == 0:
                    print(f"  Completed {i + 1}/{total_trips} trips")

            except Exception as e:
                print(f"Error on trip {i + 1} (planet {planet_name}): {e}")
                break

        df = pd.DataFrame(trips)

        if len(df) > 0:
            print("\n" + "="*60)
            print("SUMMARY OF ALTERNATE PLANET EXPLORATION")
            print("="*60)

            summary = df.groupby('planet_name')['survived'].mean() * 100
            for planet_name, survival_rate in summary.items():
                total_trips = len(df[df['planet_name'] == planet_name])
                print(f"{planet_name}: Survival Rate = {survival_rate:.2f}% over {total_trips} trips")

        return df
    
    def explore_ucb(self, total_trips: int = 90, morty_count: int = 1, exploration_weight: float = 2.0) -> pd.DataFrame:
        """
        Use the Upper Confidence Bound (UCB) strategy to choose which planet to explore.
        Balances exploration (trying less-tested planets) and exploitation (using best-performing ones).

        Args:
            total_trips: Total number of trips to make
            morty_count: Number of Morties per trip (1-3)
            exploration_weight: Multiplier for the exploration term (higher = more exploration)

        Returns:
            DataFrame with all trip data
        """
        import math

        print("\n" + "="*60)
        print("UCB PLANET EXPLORATION STRATEGY")
        print("="*60)

        self.client.start_episode()
        self.trips_data = []

        num_planets = 3
        successes = [0] * num_planets  # number of successful trips per planet
        counts = [0] * num_planets     # number of trips per planet

        trips = []

        for i in range(total_trips):
            # --- Step 1: Choose planet based on UCB formula ---
            ucb_values = []
            for p in range(num_planets):
                if counts[p] == 0:
                    # Encourage exploration of untested planets
                    ucb = float("inf")
                else:
                    avg_success = successes[p] / counts[p]
                    confidence = exploration_weight * math.sqrt(math.log(i + 1) / counts[p])
                    ucb = avg_success + confidence
                ucb_values.append(ucb)

            planet = int(np.argmax(ucb_values))
            planet_name = self.client.get_planet_name(planet)

            # --- Step 2: Send Morties to selected planet ---
            try:
                result = self.client.send_morties(planet, morty_count)

                survived = 1 if result["survived"] else 0
                counts[planet] += 1
                successes[planet] += survived

                trip_data = {
                    "trip_number": i + 1,
                    "planet": planet,
                    "planet_name": planet_name,
                    "morties_sent": result["morties_sent"],
                    "survived": survived,
                    "steps_taken": result["steps_taken"],
                    "morties_in_citadel": result["morties_in_citadel"],
                    "morties_on_planet_jessica": result["morties_on_planet_jessica"],
                    "morties_lost": result["morties_lost"],
                }

                trips.append(trip_data)
                self.trips_data.append(trip_data)

                if (i + 1) % 10 == 0:
                    print(f"  Completed {i + 1}/{total_trips} trips")

            except Exception as e:
                print(f"Error on trip {i + 1} (planet {planet_name}): {e}")
                break

        df = pd.DataFrame(trips)

        # --- Step 3: Summarize results ---
        if len(df) > 0:
            print("\n" + "="*60)
            print("SUMMARY OF UCB EXPLORATION")
            print("="*60)

            for p in range(num_planets):
                name = self.client.get_planet_name(p)
                if counts[p] > 0:
                    rate = successes[p] / counts[p] * 100
                    print(f"{name}: {counts[p]} trips, {rate:.2f}% survival rate")
                else:
                    print(f"{name}: Not explored")

        return df
    def explore_ucb_adaptive(
        self,
        total_trips: int = 90,
        morty_count: int = 1,
        exploration_weight: float = 2.0,
        memory_window: int = 20,
    ) -> pd.DataFrame:
        """
        Adaptive UCB exploration with short-term memory.
        Only recent results (last `memory_window` trips per planet) affect decisions.

        Args:
            total_trips: Total number of trips to make
            morty_count: Number of Morties per trip (1-3)
            exploration_weight: Exploration term weight (higher favors exploration)
            memory_window: Number of recent trips to remember for each planet

        Returns:
            DataFrame with all trip data
        """
        import math
        from collections import deque

        print("\n" + "=" * 60)
        print("ADAPTIVE (SHORT-MEMORY) UCB EXPLORATION")
        print("=" * 60)

        self.client.start_episode()
        self.trips_data = []

        num_planets = 3
        recent_results = [deque(maxlen=memory_window) for _ in range(num_planets)]
        trips = []

        for i in range(total_trips):
            ucb_values = []
            total_recent = sum(len(r) for r in recent_results) or 1  # avoid div by 0

            for p in range(num_planets):
                if len(recent_results[p]) == 0:
                    # Encourage exploration for planets not yet visited recently
                    ucb = float("inf")
                else:
                    avg_success = np.mean(recent_results[p])
                    confidence = exploration_weight * math.sqrt(
                        math.log(total_recent) / len(recent_results[p])
                    )
                    ucb = avg_success + confidence
                ucb_values.append(ucb)

            planet = int(np.argmax(ucb_values))
            planet_name = self.client.get_planet_name(planet)

            # --- Send Morties and record result ---
            try:
                result = self.client.send_morties(planet, morty_count)
                survived = 1 if result["survived"] else 0

                recent_results[planet].append(survived)

                trip_data = {
                    "trip_number": i + 1,
                    "planet": planet,
                    "planet_name": planet_name,
                    "morties_sent": result["morties_sent"],
                    "survived": survived,
                    "steps_taken": result["steps_taken"],
                    "morties_in_citadel": result["morties_in_citadel"],
                    "morties_on_planet_jessica": result["morties_on_planet_jessica"],
                    "morties_lost": result["morties_lost"],
                }

                trips.append(trip_data)
                self.trips_data.append(trip_data)

                if (i + 1) % 10 == 0:
                    print(f"  Completed {i + 1}/{total_trips} trips")

            except Exception as e:
                print(f"Error on trip {i + 1} (planet {planet_name}): {e}")
                break

        df = pd.DataFrame(trips)

        # --- Summary ---
        if len(df) > 0:
            print("\n" + "=" * 60)
            print("SUMMARY OF SHORT-MEMORY UCB EXPLORATION")
            print("=" * 60)

            for p in range(num_planets):
                name = self.client.get_planet_name(p)
                data = [int(x) for x in recent_results[p]]
                if data:
                    rate = np.mean(data) * 100
                    print(f"{name}: {len(data)} recent trips, {rate:.2f}% survival (last {memory_window})")
                else:
                    print(f"{name}: Not explored recently")

        return df
    def explore_ucb_adaptive_confident(
        self,
        total_trips: int = 90,
        exploration_weight: float = 2.0,
        memory_window: int = 20,
    ) -> pd.DataFrame:
        """
        Adaptive UCB exploration with short-term memory and dynamic Morty allocation.
        - Explore with 1 Morty.
        - Exploit with 2 Morties.
        - Go all-in (3 Morties) when very confident in a planet's safety.

        Args:
            total_trips: Total number of trips to make
            exploration_weight: Exploration term weight (higher favors exploration)
            memory_window: Number of recent trips to remember for each planet

        Returns:
            DataFrame with all trip data
        """
        import math
        from collections import deque

        print("\n" + "=" * 60)
        print("ADAPTIVE (SHORT-MEMORY) UCB EXPLORATION WITH DYNAMIC MORTIES")
        print("=" * 60)

        self.client.start_episode()
        self.trips_data = []

        num_planets = 3
        recent_results = [deque(maxlen=memory_window) for _ in range(num_planets)]
        trips = []

        for i in range(total_trips):
            ucb_values = []
            total_recent = sum(len(r) for r in recent_results) or 1  # avoid div by 0

            for p in range(num_planets):
                if len(recent_results[p]) == 0:
                    # Encourage exploration for unvisited planets
                    ucb = float("inf")
                else:
                    avg_success = np.mean(recent_results[p])
                    confidence = exploration_weight * math.sqrt(
                        math.log(total_recent) / len(recent_results[p])
                    )
                    ucb = avg_success + confidence
                ucb_values.append(ucb)

            planet = int(np.argmax(ucb_values))
            planet_name = self.client.get_planet_name(planet)

            # Compute confidence & estimated survival for chosen planet
            if len(recent_results[planet]) == 0:
                avg_success = 0.5  # neutral start
                confidence = float("inf")
            else:
                avg_success = np.mean(recent_results[planet])
                confidence = exploration_weight * math.sqrt(
                    math.log(total_recent) / len(recent_results[planet])
                )

            # --- Dynamic Morty allocation ---
            # More uncertainty → fewer Morties
            if confidence > 0.8:
                morty_count = 1  # still exploring
            elif avg_success > 0.7 and confidence < 0.5:
                morty_count = 3  # very confident
            else:
                morty_count = 2  # moderate exploitation

            # --- Send Morties and record result ---
            try:
                result = self.client.send_morties(planet, morty_count)
                survived = 1 if result["survived"] else 0

                recent_results[planet].append(survived)

                trip_data = {
                    "trip_number": i + 1,
                    "planet": planet,
                    "planet_name": planet_name,
                    "morties_sent": morty_count,
                    "survived": survived,
                    "steps_taken": result["steps_taken"],
                    "morties_in_citadel": result["morties_in_citadel"],
                    "morties_on_planet_jessica": result["morties_on_planet_jessica"],
                    "morties_lost": result["morties_lost"],
                    "confidence": confidence,
                    "avg_success": avg_success,
                }

                trips.append(trip_data)
                self.trips_data.append(trip_data)

                if (i + 1) % 10 == 0:
                    print(f"  Completed {i + 1}/{total_trips} trips")

            except Exception as e:
                print(f"Error on trip {i + 1} (planet {planet_name}): {e}")
                break

        df = pd.DataFrame(trips)

        # --- Summary ---
        if len(df) > 0:
            print("\n" + "=" * 60)
            print("SUMMARY OF ADAPTIVE UCB EXPLORATION")
            print("=" * 60)

            for p in range(num_planets):
                name = self.client.get_planet_name(p)
                data = [int(x) for x in recent_results[p]]
                if data:
                    rate = np.mean(data) * 100
                    print(f"{name}: {len(data)} recent trips, {rate:.2f}% survival (last {memory_window})")
                else:
                    print(f"{name}: Not explored recently")

        return df





if __name__ == "__main__":
    # Example usage
    from api_client import SphinxAPIClient
    
    try:
        client = SphinxAPIClient()
        collector = DataCollector(client)
        
        print("Data Collector initialized!")
        print("\nTo explore planets, run:")
        print("  df = collector.explore_all_planets(trips_per_planet=30)")
        print("\nOr explore a single planet:")
        print("  df = collector.explore_planet(planet=0, num_trips=50)")
        
    except Exception as e:
        print(f"Error: {e}")
