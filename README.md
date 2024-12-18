# KLTN
## Kết quả training
### Tabular IOT 2024
**LSTM**
Log folder: logs/20241216_231335
Model: LSTMModel(
  (embedding): CustomEmbedding()
  (lstm): LSTM(128, 64, num_layers=2, batch_first=True, dropout=0.2)
  (fc): Linear(in_features=64, out_features=5, bias=True)
)
Train Accuracy: 99.54%
Validation Accuracy: 99.54%
Test Accuracy: 99.54%
                            precision    recall  f1-score   support

         Benign Traffic     0.9432    0.9805    0.9615      6524
          DoS TCP Flood     0.9997    0.9995    0.9996    421384
MQTT DDoS Publish Flood     0.9715    0.9977    0.9844     82782
 MQTT DoS Connect Flood     0.9975    0.9496    0.9730     47606
        Recon Port Scan     0.9999    0.9988    0.9993     97105