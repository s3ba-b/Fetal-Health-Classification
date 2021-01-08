using Microsoft.ML.Data;

namespace MulticlassClassification_Fetal.DataStructures
{
    public class FetalHealthData
    {
        [LoadColumn(0)]
        public float BaselineValue { get; set; }
        [LoadColumn(1)]
        public float Accelerations { get; set; }
        [LoadColumn(2)]
        public float FetalMovement { get; set; }
        [LoadColumn(3)]
        public float UterineContractions { get; set; }
        [LoadColumn(4)]
        public float LightDecelerations { get; set; }
        [LoadColumn(5)]
        public float SevereDecelerations { get; set; }
        [LoadColumn(6)]
        public float ProlonguedDecelerations { get; set; }
        [LoadColumn(7)]
        public float AbnormalShortTermVariability { get; set; }
        [LoadColumn(8)]
        public float MeanValueOfShortTermVariability { get; set; }
        [LoadColumn(9)]
        public float PercentageOfTimeWithAbnormalLongTermVariability { get; set; }
        [LoadColumn(10)]
        public float MeanValueOfLongTermVariability { get; set; }
        [LoadColumn(11)]
        public float HistogramWidth { get; set; }
        [LoadColumn(12)]
        public float HistogramMin { get; set; }
        [LoadColumn(13)]
        public float HistogramMax { get; set; }
        [LoadColumn(14)]
        public float HistogramNumberOfPeaks { get; set; }
        [LoadColumn(15)]
        public float HistogramNumberOfZeroes { get; set; }
        [LoadColumn(16)]
        public float HistogramMode { get; set; }
        [LoadColumn(17)]
        public float HistogramMean { get; set; }
        [LoadColumn(18)]
        public float HistogramMedian { get; set; }
        [LoadColumn(19)]
        public float HistogramVariance { get; set; }
        [LoadColumn(20)]
        public float HistogramTendency { get; set; }
        [LoadColumn(21)]
        public float FetalHealth { get; set; }
    }
}