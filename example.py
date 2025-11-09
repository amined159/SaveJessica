"""
Example script demonstrating how to use the Morty Express Challenge API.

This script shows how to:
1. Initialize the API client
2. Explore planets to collect data
3. Visualize the results
4. Determine the best strategy
"""

from api_client import SphinxAPIClient
from data_collector import DataCollector
from visualizations import (
    plot_survival_rates,
    plot_survival_by_planet,
    plot_moving_average,
    plot_risk_evolution,
    plot_episode_summary,
    create_all_visualizations
)


def main():
    """Main example demonstrating the workflow."""
    
    print("="*60)
    print("MORTY EXPRESS CHALLENGE - EXAMPLE SCRIPT")
    print("="*60)
    
    # Step 1: Initialize the API client
    print("\n1. Initializing API client...")
    try:
        client = SphinxAPIClient()
        print("✓ API client initialized successfully!")
    except ValueError as e:
        print(f"✗ Error: {e}")
        print("\nPlease set up your API token:")
        print("1. Visit https://challenge.sphinxhq.com/")
        print("2. Request a token with your name and email")
        print("3. Create a .env file with: SPHINX_API_TOKEN=your_token_here")
        return
    
    # Step 2: Start a new episode
    print("\n2. Starting a new episode...")
    result = client.start_episode()
    print(f"✓ Episode started!")
    print(f"   Morties in Citadel: {result['morties_in_citadel']}")
    
    # Step 3: Explore planets to collect data
    print("\n3. Collecting data from all planets...")
    collector = DataCollector(client)
    
    # Explore each planet with a smaller number of trips for the example
    # You can increase trips_per_planet for more data
    df = collector.explore_ucb_adaptive_confident(total_trips=1000, exploration_weight=2, memory_window=700)
    
    # Step 4: Analyze the data
    print("\n4. Analyzing collected data...")
    risk_analysis = collector.analyze_risk_changes(df)
    
    print("\nRisk Analysis:")
    for planet_name, data in risk_analysis.items():
        print(f"\n{planet_name}:")
        print(f"  Overall Survival Rate: {data['overall_survival_rate']:.2f}%")
        print(f"  Early Survival Rate: {data['early_survival_rate']:.2f}%")
        print(f"  Late Survival Rate: {data['late_survival_rate']:.2f}%")
        print(f"  Trend: {data['trend']} ({data['change']:+.2f}%)")
    
    # Step 5: Determine best planet
    print("\n5. Determining best planet...")
    best_planet, best_planet_name = collector.get_best_planet(df, consider_trend=True)
    print(f"✓ Best planet: {best_planet_name} (index {best_planet})")
    
    # Step 6: Save the data
    print("\n6. Saving data...")
    collector.save_data("exploration_data.csv")
    
    # Step 7: Create visualizations
    print("\n7. Creating visualizations...")
    print("\nGenerating plots...")
    
    # Individual plots
    print("\n- Survival rates over time...")
    plot_survival_rates(df)
    
    print("- Survival by planet comparison...")
    plot_survival_by_planet(df)
    
    print("- Moving average analysis...")
    plot_moving_average(df, window=10)
    
    print("- Risk evolution...")
    plot_risk_evolution(df)
    
    print("- Episode summary dashboard...")
    plot_episode_summary(df)
    
    # Optionally, save all plots
    # create_all_visualizations(df, output_dir="plots")
    
    # Step 8: Final status
    print("\n8. Final Status:")
    status = client.get_status()
    print(f"   Morties in Citadel: {status['morties_in_citadel']}")
    print(f"   Morties on Planet Jessica: {status['morties_on_planet_jessica']}")
    print(f"   Morties Lost: {status['morties_lost']}")
    print(f"   Steps Taken: {status['steps_taken']}")
    
    success_rate = (status['morties_on_planet_jessica'] / 1000) * 100
    print(f"\n   Success Rate: {success_rate:.2f}%")
    
    print("\n" + "="*60)
    print("EXAMPLE COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Analyze the visualizations to understand planet behavior")
    print("2. Implement your own strategy in strategy.py")
    print("3. Run your strategy to maximize saved Morties!")
    print("\nGood luck saving Jessica! 🚀")


def quick_explore():
    """Quick exploration of a single planet."""
    
    print("Quick Planet Exploration")
    print("="*40)
    
    try:
        client = SphinxAPIClient()
        collector = DataCollector(client)
        
        # Start new episode
        client.start_episode()
        
        # Explore Planet A (index 0) with 50 trips
        print("\nExploring 'On a Cob' Planet...")
        df = collector.explore_planet(planet=0, num_trips=50, morty_count=1)
        
        # Show simple visualization
        plot_survival_rates(df)
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Run the main example
    main()
    
    # Or run a quick exploration of a single planet
    # quick_explore()
