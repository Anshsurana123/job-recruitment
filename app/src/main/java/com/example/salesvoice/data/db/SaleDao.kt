package com.example.salesvoice.data.db

import androidx.lifecycle.LiveData
import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import com.example.salesvoice.data.model.DailyTotals
import com.example.salesvoice.data.model.SaleEntry

@Dao
interface SaleDao {
    @Query("SELECT * FROM sales WHERE date(timestamp/1000, 'unixepoch', 'localtime') = date('now', 'localtime') ORDER BY timestamp DESC")
    fun getTodaySales(): LiveData<List<SaleEntry>>

    @Query("SELECT * FROM sales WHERE date(timestamp/1000, 'unixepoch', 'localtime') = :date ORDER BY timestamp DESC")
    fun getSalesForDate(date: String): LiveData<List<SaleEntry>>   // date format: YYYY-MM-DD

    @Query("SELECT SUM(totalAmount) as totalRevenue, SUM(totalProfit) as totalProfit FROM sales WHERE date(timestamp/1000, 'unixepoch', 'localtime') = date('now', 'localtime')")
    fun getTodayTotals(): LiveData<DailyTotals>

    @Query("SELECT SUM(totalAmount) as totalRevenue, SUM(totalProfit) as totalProfit FROM sales WHERE date(timestamp/1000, 'unixepoch', 'localtime') = :date")
    fun getTotalsForDate(date: String): LiveData<DailyTotals>

    @Insert suspend fun insert(sale: SaleEntry): Long

    @Query("DELETE FROM sales WHERE id = :saleId")
    suspend fun delete(saleId: Long)
}
