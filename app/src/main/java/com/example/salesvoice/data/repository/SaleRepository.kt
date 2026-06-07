package com.example.salesvoice.data.repository

import androidx.lifecycle.LiveData
import com.example.salesvoice.data.db.SaleDao
import com.example.salesvoice.data.model.DailyTotals
import com.example.salesvoice.data.model.SaleEntry
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class SaleRepository(private val saleDao: SaleDao) {

    fun getTodaySales(): LiveData<List<SaleEntry>> = saleDao.getTodaySales()

    fun getSalesForDate(date: String): LiveData<List<SaleEntry>> = saleDao.getSalesForDate(date)

    fun getTodayTotals(): LiveData<DailyTotals> = saleDao.getTodayTotals()

    fun getTotalsForDate(date: String): LiveData<DailyTotals> = saleDao.getTotalsForDate(date)

    suspend fun insertSale(sale: SaleEntry): Long = withContext(Dispatchers.IO) {
        saleDao.insert(sale)
    }

    suspend fun deleteSale(saleId: Long) = withContext(Dispatchers.IO) {
        saleDao.delete(saleId)
    }
}
