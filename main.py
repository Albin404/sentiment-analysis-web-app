"""Simple menu for training or predicting sentiment."""
from predict import predict_sentiment
from train import train_model


def main():
    print("=== Sentiment Analysis Mini Project ===")
    while True:
        print("\n1. Train model\n2. Predict sentiment\n3. Exit")
        choice = input("Choose an option (1-3): ").strip()
        if choice == "1":
            train_model()
        elif choice == "2":
            review = input("Enter a review: ").strip()
            if review:
                try:
                    print("Predicted sentiment:", predict_sentiment(review).title())
                except FileNotFoundError as error:
                    print(error)
            else:
                print("Review cannot be empty.")
        elif choice == "3":
            print("Thank you. Goodbye!")
            break
        else:
            print("Invalid option. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
