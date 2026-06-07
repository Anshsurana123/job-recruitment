package com.example.salesvoice.data.model

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "sales")
data class SaleEntry(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val productId: Long,
    val productName: String,
    val quantity: Double,
    val unit: String,
    val pricePerUnit: Double,
    val profitPerUnit: Double,
    val totalAmount: Double,
    val totalProfit: Double,
    val timestamp: Long = System.currentTimeMillis()
)
