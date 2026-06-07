package com.example.salesvoice.data.db

import androidx.lifecycle.LiveData
import androidx.room.*
import com.example.salesvoice.data.model.Product

@Dao
interface ProductDao {
    @Query("SELECT * FROM products ORDER BY name ASC")
    fun getAllProducts(): LiveData<List<Product>>

    @Query("SELECT * FROM products")
    suspend fun getAllProductsSync(): List<Product>

    @Insert suspend fun insert(product: Product): Long
    @Update suspend fun update(product: Product)
    @Delete suspend fun delete(product: Product)
}
