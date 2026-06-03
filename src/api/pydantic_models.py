from pydantic import BaseModel, Field, ConfigDict

class PredictionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Recency": 5.0,
                "Frequency": 14.0,
                "Total_Spending": 125000.0,
                "Average_Transaction_Value": 8928.5,
                "Max_Transaction_Value": 35000.0,
                "Transaction_Value_Std": 2100.0
            }
        }
    )
    Recency: float = Field(..., description="Days elapsed since the customer's last transaction")
    Frequency: float = Field(..., description="Total platform transaction count executed by the user")
    Total_Spending: float = Field(..., description="Absolute monetary footprint volume")
    Average_Transaction_Value: float = Field(..., description="Mean ticket size per transaction")
    Max_Transaction_Value: float = Field(..., description="Peak observed transaction size spike")
    Transaction_Value_Std: float = Field(..., description="Standard deviation tracking spending variance")

class PredictionResponse(BaseModel):
    Credit_Risk_Assessment: str = Field(..., description="Final approval output ('APPROVED' or 'DENIED')")
    Risk_Classification_Label: int = Field(..., description="Binary category classification (0 for Good, 1 for High Risk)")
    Default_Probability_Score: float = Field(..., description="Calculated probability score of loan default")
    Operational_Message: str = Field(..., description="Status info message matching compliance tracking")