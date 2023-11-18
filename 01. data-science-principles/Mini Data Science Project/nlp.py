# import pandas as pd
# from transformers import BertTokenizer, BertForSequenceClassification
# from torch.utils.data import DataLoader, TensorDataset
# import torch
# from sklearn.preprocessing import LabelEncoder


# def train_bert_model(
#     labeled_file, unlabeled_file, text_column, label_column=None, output_csv=None
# ):
#     # Load labeled data
#     labeled_df = pd.read_csv(labeled_file)

#     # Assuming your labeled data has 'text' and 'label' columns
#     labeled_df = labeled_df.dropna(subset=[text_column, label_column])
#     X_labeled = labeled_df[text_column]
#     y_labeled = labeled_df[label_column]

#     # Convert labels to numeric values using LabelEncoder
#     label_encoder = LabelEncoder()
#     y_labeled = label_encoder.fit_transform(y_labeled)

#     print("Encoding finished")

#     # Load pre-trained BERT model and tokenizer
#     model_name = "bert-base-uncased"
#     tokenizer = BertTokenizer.from_pretrained(model_name)
#     model = BertForSequenceClassification.from_pretrained(model_name)

#     print("Model loaded")

#     # Tokenize and encode the labeled data
#     tokenized_data = tokenizer(
#         X_labeled.tolist(), padding=True, truncation=True, return_tensors="pt"
#     )
#     input_ids = tokenized_data["input_ids"]
#     attention_mask = tokenized_data["attention_mask"]

#     print("Data tokenized, starting training")

#     # Train the model on labeled data
#     train_dataset = TensorDataset(
#         input_ids, attention_mask, torch.tensor(y_labeled, dtype=torch.long)
#     )  # Ensure dtype is long
#     # train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

#     # optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

#     model.train()

#     print("model training")

#     # for epoch in range(3):  # You might need to adjust the number of epochs
#     #     for batch in train_loader:
#     #         optimizer.zero_grad()
#     #         input_ids, attention_mask, labels = batch
#     #         outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#     #         loss = outputs.loss
#     #         loss.backward()
#     #         optimizer.step()

#     print("loop finished")

#     # Load unlabeled data
#     unlabeled_df = pd.read_csv(unlabeled_file)

#     print("unlabeled df  imported")

#     # Drop rows with null values in the specified text column
#     unlabeled_df = unlabeled_df.dropna(subset=[text_column])
#     X_unlabeled = unlabeled_df[text_column]

#     print("df cleaned, starting encoding unlabeled data")

#     # Tokenize and encode the unlabeled data
#     tokenized_unlabeled = tokenizer(
#         X_unlabeled.tolist(), padding=True, truncation=True, return_tensors="pt"
#     )
#     input_ids_unlabeled = tokenized_unlabeled["input_ids"]
#     attention_mask_unlabeled = tokenized_unlabeled["attention_mask"]

#     print("unlabeled data tokenized, prediction started")

#     # Predict labels for unlabeled data
#     model.eval()
#     with torch.no_grad():
#         logits_unlabeled = model(
#             input_ids_unlabeled, attention_mask=attention_mask_unlabeled
#         ).logits

#     predicted_labels = torch.argmax(logits_unlabeled, dim=1)

#     print("labels predicted")

#     # Decode predicted labels back to original labels
#     predicted_labels = label_encoder.inverse_transform(predicted_labels.numpy())

#     # Add predicted labels to the unlabeled data
#     unlabeled_df["predicted_label"] = predicted_labels

#     print("labels added to df")

#     # Save predictions to CSV if specified
#     if output_csv:
#         unlabeled_df.to_csv(output_csv, index=False)

#     return unlabeled_df


# # Example usage:
# labeled_file_path = "labeled_data.csv"
# unlabeled_file_path = "unlabeled_data.csv"
# output_csv_path = "predicted_data.csv"

# # Assuming your labeled data has columns 'text' and 'label'
# # If you have a different label column name, replace 'label_column' accordingly
# predicted_df = train_bert_model(
#     labeled_file_path,
#     unlabeled_file_path,
#     text_column="text",
#     label_column="label",
#     output_csv=output_csv_path,
# )

# # Display the predicted DataFrame
# print(predicted_df.head())
