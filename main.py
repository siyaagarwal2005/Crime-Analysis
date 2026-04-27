#!/usr/bin/env python3
"""
Crime Dataset Analysis - Main Entry Point
Automated Leaderboard - Every user's results are tracked!
"""

import os
import sys
import argparse
from colorama import init, Fore, Style
import getpass

init(autoreset=True)

from data_processor import DataProcessor
from train_models import ModelTrainer
from leaderboard_manager import LeaderboardManager
from config import MODELS

def print_banner():
    """Print welcome banner"""
    banner = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════════╗
║                                                                          ║
║   🔍 CRIME DATASET ANALYSIS & PREDICTION PLATFORM                       ║
║   📊 Automated Leaderboard - Track Your Performance!                    ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""
    print(banner)

def get_user_info():
    """Get user information for leaderboard"""
    print(f"{Fore.YELLOW}📝 Please enter your details for the leaderboard:{Style.RESET_ALL}")
    name = input("   Enter your name: ").strip()
    if not name:
        name = "Anonymous"
    
    # Optional: get email for contact
    email = input("   Enter your email (optional): ").strip()
    
    return name, email

def run_single_model():
    """Run a single model and submit to leaderboard"""
    print_banner()
    
    # Get user info
    user_name, email = get_user_info()
    
    # Initialize components
    processor = DataProcessor()
    leaderboard = LeaderboardManager()
    
    # Show leaderboard stats
    stats = leaderboard.get_statistics()
    if stats['total_submissions'] > 0:
        print(f"\n{Fore.GREEN}📊 Current Leaderboard Stats:{Style.RESET_ALL}")
        print(f"   👥 Participants: {stats['unique_users']}")
        print(f"   📝 Submissions: {stats['total_submissions']}")
        print(f"   🏆 Best F1 Score: {stats['max_f1_score']:.4f} ({stats['best_user']} - {stats['best_model']})")
    
    # Select model
    print(f"\n{Fore.YELLOW}🤖 Available Models:{Style.RESET_ALL}")
    model_list = list(MODELS.keys())
    for i, model in enumerate(model_list, 1):
        print(f"   {i}. {model}")
    
    while True:
        try:
            choice = int(input("\n   Select model (1-{}): ".format(len(model_list))))
            if 1 <= choice <= len(model_list):
                selected_model = model_list[choice - 1]
                break
            else:
                print(f"   Please enter a number between 1 and {len(model_list)}")
        except ValueError:
            print("   Please enter a valid number")
    
    # Process data
    print(f"\n{Fore.CYAN}🔄 Processing data...{Style.RESET_ALL}")
    X_train, X_test, y_train, y_test, feature_cols = processor.run_pipeline()
    
    # Initialize trainer and get model config
    trainer = ModelTrainer()
    model_config = MODELS[selected_model]
    
    # Train and evaluate selected model
    result = trainer.train_and_evaluate(
        selected_model, model_config, X_train, X_test, y_train, y_test
    )
    
    # Display results
    print(f"\n{Fore.GREEN}✅ Model Evaluation Complete!{Style.RESET_ALL}")
    print("-" * 50)
    print(f"   📊 Accuracy:  {result['accuracy']:.4f}")
    print(f"   🎯 F1 Score:  {result['f1']:.4f}")
    print(f"   📈 Precision: {result['precision']:.4f}")
    print(f"   📉 Recall:    {result['recall']:.4f}")
    print(f"   ⏱️  Time:      {result['train_time']:.2f} seconds")
    
    # Submit to leaderboard
    print(f"\n{Fore.YELLOW}📝 Submitting to leaderboard...{Style.RESET_ALL}")
    rank = leaderboard.add_entry(
        user_name, selected_model, 
        result['accuracy'], result['f1'], 
        result['precision'], result['recall']
    )
    
    print(f"{Fore.GREEN}✅ Submitted successfully!{Style.RESET_ALL}")
    if rank:
        print(f"   🏅 Your current rank: #{rank}")
    
    # Show updated leaderboard
    leaderboard.display_leaderboard()
    
    # Ask to save model
    save_model = input("\n💾 Save this model for later use? (y/n): ").lower()
    if save_model == 'y':
        import joblib
        from config import MODELS_DIR
        joblib.dump(result['model'], f"{MODELS_DIR}/user_{user_name.replace(' ', '_')}_{selected_model.replace(' ', '_')}.pkl")
        print(f"   ✅ Model saved to {MODELS_DIR}")

def run_all_models():
    """Run all models and compare"""
    print_banner()
    
    print(f"{Fore.YELLOW}🚀 Running comprehensive model comparison...{Style.RESET_ALL}")
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Run all models
    results = trainer.run_all_models()
    
    # Display results
    comparison_df = trainer.display_results()
    
    # Show leaderboard (optional - user can submit best model)
    print(f"\n{Fore.YELLOW}💡 Tip: You can submit your best model to the leaderboard by running: python main.py --single{Style.RESET_ALL}")

def view_leaderboard():
    """Display the leaderboard"""
    print_banner()
    leaderboard = LeaderboardManager()
    leaderboard.display_leaderboard()
    
    # Show statistics
    stats = leaderboard.get_statistics()
    print(f"\n{Fore.CYAN}📊 Leaderboard Statistics:{Style.RESET_ALL}")
    print(f"   Total submissions: {stats['total_submissions']}")
    print(f"   Unique users: {stats['unique_users']}")
    print(f"   Average F1 Score: {stats['avg_f1_score']:.4f}")
    print(f"   Best F1 Score: {stats['max_f1_score']:.4f}")

def view_user_history():
    """View submission history for a user"""
    print_banner()
    user_name = input("Enter username to view history: ").strip()
    leaderboard = LeaderboardManager()
    leaderboard.display_user_history(user_name)

def clear_leaderboard():
    """Clear the leaderboard (admin function)"""
    print_banner()
    print(f"{Fore.RED}⚠️  WARNING: This will delete all leaderboard entries!{Style.RESET_ALL}")
    confirm = input("Type 'CONFIRM' to proceed: ")
    if confirm == 'CONFIRM':
        leaderboard = LeaderboardManager()
        leaderboard.clear_leaderboard(confirm=True)
    else:
        print("❌ Operation cancelled")

def main():
    parser = argparse.ArgumentParser(description='Crime Dataset Analysis Platform')
    parser.add_argument('--single', action='store_true', help='Run a single model and submit to leaderboard')
    parser.add_argument('--all', action='store_true', help='Run all models for comparison')
    parser.add_argument('--leaderboard', action='store_true', help='View the leaderboard')
    parser.add_argument('--history', action='store_true', help='View user submission history')
    parser.add_argument('--clear', action='store_true', help='Clear leaderboard (admin)')
    
    args = parser.parse_args()
    
    if args.single:
        run_single_model()
    elif args.all:
        run_all_models()
    elif args.leaderboard:
        view_leaderboard()
    elif args.history:
        view_user_history()
    elif args.clear:
        clear_leaderboard()
    else:
        # Interactive menu
        print_banner()
        print(f"{Fore.YELLOW}Please choose an option:{Style.RESET_ALL}")
        print("   1. Run single model and submit to leaderboard")
        print("   2. Run all models comparison")
        print("   3. View leaderboard")
        print("   4. View user history")
        print("   5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            run_single_model()
        elif choice == '2':
            run_all_models()
        elif choice == '3':
            view_leaderboard()
        elif choice == '4':
            view_user_history()
        elif choice == '5':
            print("Goodbye!")
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()
