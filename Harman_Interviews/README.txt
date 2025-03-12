    1. Asked about fixed point implementation of area of a circle for 16bit registers processors.
    2. How would you implement 16x16 multiplication using 16bit registers.
    3. You have only FFT function how would you implement IFFT??
    4. Explain the block diagram of sampling rate conversion.
    5. What are the constraints on M and L, the down sample and up sample factors. There should not be any common factors. And they should be integers. He was expecting something else also, needs to be explored.
    6. What is difference between asynchronous and synchronous transmission. The answer that I gave is in asynchronous transmission interrupts signal the completion of task. And in synchronous the resource expects the data to be ready by some predetermined time.
    7. Next he asked about SPI and I2C protocols.
    8. How will you count the number of zero crossing in signal. And how would you determine if signal is periodic or not by the number of zero crossings.
    9. There is signal which is sampled at 8khz. We need to identify if 1khz signal is present in this signal. What is minimum length of samples that one need to consider. I told since 1khz to get one complete cycle we need atleast 1ms seconds of data, so anything above 1ms we will be able to detect the 1khz. But there is another aspect to it which is the frequency should fall in one of the bin of sampled DTFT, we it spans more than one bin then there will be two peaks then we will not be able to detect the 1khz signal.


10th Oct 2024
    1. Asked about how would you decide which filter do you select FIR or IIR for a given application.
    2. 30khz needs to be down sampled to 10khz. The highest frequency component is 10khz in signal. I said drop every two samples. But I should have said that we need to LPF so that aliasing doesn't occur and the drop two samples. Here I didn't consider that if 10khz is highest frequency component then one needs to sample at 20khz, so if we want to down sample to 10khz then we need to LPF with 5khz cutoff and then drop two samples. This was huge mistake, by giving the highest frequency component and new sampling rate same value he confused me.
    3. There is filter of order 80 and there are 100 samples. Then how would you optimize it.
    4. What are the C optimization techniques that you have used.
    5. What is L1 cache, how is data loaded into the L1 cache from the external memory. We modulo with the address and then place at the corresponding cache line location. And then maintain a mapping. This what I told.
    6. If there are lot of cache misses are happening then how will handle the situation. This I couldn't answer.
    7. In IIR filter implementation which is better DF I or DF II. I said both are same, need to check this.
    8. Why do we round while converting floating point to fixed point. I couldn't give a convincing answer. Need to check this.
    9. How would you identify if stalls are there in your code and how would you remove them?? I said in RVDS we can see how many cycles each instruction is taking and based on that adjust the instructions order(position) so that stalls are removed.
    10. Do you know the IEEE standard for representing float values??This needs to explored.
    11. If I have FIR and IIR filter of same order. Then which one will have more computations. I asked if there is numerator term also present in the IIR filter. I should have also told(but didn't tell this) that computationally FIR would be faster if we have SIMD, because for IIR we need to wait for each output to be generated before starting the next iteration. So IIR would be slower even if we have SIMD capability on that processor.
    12. Why would you need more coefficients for FIR filter and less for IIR filter for the same frequency response. This needs to be explored.
    
    
