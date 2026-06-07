package com.example.salesvoice.data.model

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(
    tableName = "products",
    indices = [Index(value = ["name"], unique = true)]
)
data class Product(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val name: String,
    val aliases: String,        // stored as comma-separated string, parsed on read
    val unit: String,
    val pricePerUnit: Double,
    val profitPerUnit: Double,
    val createdAt: Long = System.currentTimeMillis()
) {
    fun getAliasList(): List<String> {
        return aliases.split(",").map { it.trim() }.filter { it.isNotBlank() }
    }
}
